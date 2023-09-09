import numpy as np
from scipy.integrate import solve_ivp

import gym
from gym import spaces

from src.rl_util import load_healthy_data
from src.sb3_BA_ode import BA_ode

class psc_ba_env(gym.Env):
    def __init__(self, adaptation_duration, data_ID, max_ba_flow):
        super(psc_ba_env).__init__()

        init_state, init_param, PSC_data = load_healthy_data(max_ba_flow)

        self.action_var_names = {0:"synthesis", 1:"syn_frac_CA", 2:"hep_extract_ratio_conj_tri", 3:"hep_extract_ratio_conj_di", 4:"max_asbt_rate"}
        self.action_param_pos_mapping = {0:0, 1:1, 2:8, 3:9, 4:15} # {i:j} [action_vector][i] maps to [param_vector][j]
        self.action_original = {0:0.8236344, 1:0.5300548, 2:0.9520841, 3:0.8383059, 4:0.0436090}
        self.action_amplitude = {0:0.1, 1:0.05, 2:0.05, 3:0.05, 4:0.1} # 0,4: mulplicative; 1,2,3: additive
        
        self.init_state = init_state
        self.init_param = init_param
        
        self.PSC_log_data = PSC_data[f"PSC_log10_{data_ID}"].values
        self.PSC_log_std = PSC_data["PSC_log10_std"].values
        self.PSC_log_max = PSC_data["PSC_log10_max"].values
        
        ############### SETUP ODE PARAMS ###############
        dt = 1
        N_STEP = int(adaptation_duration / dt) + 1
        TIMEPOINTS = np.linspace(0, adaptation_duration, N_STEP)
        step_per_round = 1440

        t_to_solve = TIMEPOINTS.copy()
        self.t_span_rk45 = []
        self.t_eval_rk45 = []
        self.n_round = 0
        while len(t_to_solve) > 1:
            self.n_round += 1
            len_to_solve = min(step_per_round + 1, len(t_to_solve))
            self.t_span_rk45.append((t_to_solve[0], t_to_solve[len_to_solve-1]))
            self.t_eval_rk45.append(t_to_solve[:len_to_solve])
            t_to_solve = t_to_solve[(len_to_solve-1):]
        
        ############### SETUP CONSTANTS ###############
        self.N_STATE = len(init_state)
        self.N_ACTION_VAR = len(self.action_var_names)

        self.BAD_END_PENALTY_VIOLATE_BOUNDARY = 100
        self.PER_STEP_REWARD = 1
        self.FITTING_ERROR_CAP = 20
        self.PARAM_DEVIATION_CAP = 10

        self.boundary_low  = np.array([0, 0, 0,
                                       0, 0, 0,
                                       0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0,
                                       0, 0, 0, 0, 0, 0,
                                       0.1, 0.2, 0.01, 0.01, 0.004], dtype=np.float32) + 1e-9

        self.boundary_high = np.hstack((np.array([4000, 4000, 4000,
                                                  6000, 6000, 6000,
                                                  2000, 2000, 2000, 200,  200,  200,
                                                  2000, 2000, 2000, 2000, 2000, 4000]),
                                        np.power(10, PSC_data["PSC_log10_max"].values) * 1.5,
                                        np.array([5, 0.8, 0.999, 0.999, 0.4]))).astype(np.float32)

        self.NORMAL_CHOLESTEROL_ELIMINATION = 0.8236344 * 1440
        self.NORMAL_DIGESTION = 1025099.2233741395 # daily small intestine bile acid exposure in healthy controls: rk45_sol["original"][12:18,-1440:].sum()
        self.NORMAL_LIVER_BA = 394111.19908590394 # daily liver bile acid exposure in healthy controls: rk45_sol["original"][:3,-1440:].sum()
        self.MAX_TOXICITY = self.boundary_high[:3].sum() * step_per_round
        
        ############### RESET RL ENVIRONMENT ###############
        _ = self.reset()
        self.action_space = spaces.Discrete(3**self.N_ACTION_VAR)
        self.observation_space = spaces.Box(low=self.boundary_low * 0.2, high=self.boundary_high * 5)
    
    def reset(self):
        self.round = 0
        
        self.current_state = np.zeros(self.N_STATE + self.N_ACTION_VAR, dtype=float)
        self.current_state[:self.N_STATE] = self.init_state # ODE state
        
        self.param = self.init_param.copy()
        for key, item in self.action_param_pos_mapping.items():
            self.current_state[self.N_STATE + key] = self.param[item]

        return self.current_state

    def violate_boundary(self):
        for i in range(self.N_STATE):
            if (self.current_state[i] < self.boundary_low[i]) or (self.current_state[i] > self.boundary_high[i]):
                return True
        return False
    
    def quantify_cholesterol_elimination(self):
        return min(self.current_state[self.N_STATE] * 1440 / self.NORMAL_CHOLESTEROL_ELIMINATION, 1)

    def quantify_toxicity(self, ode_traj):
        return max((ode_traj[:3, :].sum() - self.NORMAL_LIVER_BA) / self.MAX_TOXICITY, 0)
    
    def quantify_digestion(self, ode_traj):
        return min(ode_traj[6:12, :].sum() / self.NORMAL_DIGESTION, 1)
    
    def quantify_fitting_error(self, model_estimation):
        model_pl = model_estimation[-6:]
        res = (np.log10(model_pl) - self.PSC_log_data) / self.PSC_log_std
        return min(np.sum(res**2), self.FITTING_ERROR_CAP)
    
    def quantify_param_deviation(self):
        '''
        For each param, deviation quantify the minimum steps it take to go from original value to current value
        For example, synthesis = 0.823 originally, now it is 0.1
        Then it takes (np.log(0.1) - np.log(0.823)) / np.log(1-0.25) = 7.32678 steps to get there
        See that 0.823 * ((1-0.25)**7.32678) = 0.100000276
        Roughly each param is allowed to change at most 10 steps, so deviation is bounded by 10 * 5 params = 50
        '''
        deviation = 0
        
        for key, item in self.action_var_names.items():
            action_value = self.current_state[self.N_STATE + key]
            action_change_sign = 1 if action_value >= self.action_original[key] else -1
            if item in ["synthesis", "max_asbt_rate"]:
                deviation += min((np.log(action_value) - np.log(self.action_original[key])) / np.log(1 + action_change_sign * self.action_amplitude[key]), self.PARAM_DEVIATION_CAP)
            elif item in ["syn_frac_CA", "hep_extract_ratio_conj_tri", "hep_extract_ratio_conj_di"]:
                deviation += min((action_value - self.action_original[key]) / (action_change_sign * self.action_amplitude[key]), self.PARAM_DEVIATION_CAP)
        
        return deviation

    def generate_reward(self, ode_traj, ode_integrate_SUCCESS):
        
        if self.violate_boundary() or (not ode_integrate_SUCCESS):
            reward = - self.BAD_END_PENALTY_VIOLATE_BOUNDARY
            done = True
            
            cholesterol_elimination, toxicity, digestion, fitting_error, param_deviation = np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            reward = self.PER_STEP_REWARD * 4
            
            cholesterol_elimination = self.quantify_cholesterol_elimination()
            toxicity = self.quantify_toxicity(ode_traj)
            digestion = self.quantify_digestion(ode_traj)
            fitting_error = self.quantify_fitting_error(ode_traj[:,-1])
            param_deviation = self.quantify_param_deviation()
            
            reward += cholesterol_elimination
            reward -= toxicity
            reward += digestion
            reward -= param_deviation * 0.02
            
            if self.round > 28:
                reward -= fitting_error * 0.2

            if self.round == self.n_round:
                done = True
            else:  
                done = False
            
        return reward, done, cholesterol_elimination, toxicity, digestion, fitting_error, param_deviation
    
    def convert_action_idx_to_tuple(self, action):
        '''
        **action** $\in$ {0,3^len(actions)-1} from [-1,...,-1] to [1,...,1]
        - action = random.randint(0, 3**N_ACTION_VAR-1) # random strategy 
        - action = int((3**N_ACTION_VAR-1)/2) # no change strategy: actions = [0,...,0]
        '''
        actions = []
        for i in range(self.N_ACTION_VAR):
            action, a = np.divmod(action, 3)
            actions.append(a-1)
        actions = np.array(actions, dtype=np.int64)
        return actions
    
    def update_rl_state_with_action(self, action):
        actions = self.convert_action_idx_to_tuple(action)     
        
        for key, item in self.action_var_names.items():
            # update the (RL) state vector (action var part) with new ODE param            
            if item in ["synthesis", "max_asbt_rate"]:
                self.current_state[self.N_STATE + key] *= (1 + self.action_amplitude[key] * actions[key])
            elif item in ["syn_frac_CA", "hep_extract_ratio_conj_tri", "hep_extract_ratio_conj_di"]:
                self.current_state[self.N_STATE + key] += self.action_amplitude[key] * actions[key]
            
            # cap the parameter value at their upper and lower bounds
            self.current_state[self.N_STATE + key] = np.clip(self.current_state[self.N_STATE + key],
                                                             a_min=self.boundary_low[self.N_STATE + key] + 1e-9,
                                                             a_max=self.boundary_high[self.N_STATE + key] - 1e-9)
                
            # reflect the update on the actual ODE param variable
            self.param[self.action_param_pos_mapping[key]] = self.current_state[self.N_STATE + key]
    
    def step(self, action):
        '''
        Called at each step to update its state based on the given action
        '''
        # Update the values of adaptive parameters (part of state vector) based on the action taken
        self.update_rl_state_with_action(action)
        
        # Check if any physiological ranges are violated        
        if self.violate_boundary():
            # If violated, set a negative reward and end the episode
            reward = - self.BAD_END_PENALTY_VIOLATE_BOUNDARY
            done = True
            
            return np.array([np.nan]), reward, done, {"ODE_time":np.nan, "ODE_traj":np.array([np.nan]),
                                                      "toxicity":np.nan, "digestion":np.nan, "fitting_error":np.nan, "param_deviation":np.nan}
        else:
            # Otherwise, interact with the environment through running the ODE for one step (one-day) with the updated parameters
            sol = solve_ivp(BA_ode, 
                            t_span = self.t_span_rk45[self.round],
                            y0 = self.current_state[:self.N_STATE],
                            t_eval = self.t_eval_rk45[self.round],
                            args = (self.param,))

            # Update the values of bile acids (part of RL state vector) with signals from the environment (ODE solution after one step)
            self.current_state = np.hstack(((sol.y)[:,-1], self.current_state[self.N_STATE:]))
            self.round += 1

            # Calculate the reward terms with signals from the environment (ODE trajectory)
            reward, done, cholesterol_elimination, toxicity, digestion, fitting_error, param_deviation = self.generate_reward(sol.y, sol.success)
            
            return self.current_state, reward, done, {"ODE_time":self.t_span_rk45[self.round-1][1], "ODE_traj":sol.y,
                                                      "cholesterol_elimination":cholesterol_elimination, "toxicity":toxicity, "digestion":digestion, "fitting_error":fitting_error, "param_deviation":param_deviation}