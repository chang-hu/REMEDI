import numpy as np
import pandas as pd

from stable_baselines3.common.callbacks import BaseCallback

def load_healthy_data(max_ba_flow):

    # Load healthy bile acid states

    sips_state_stable = pd.read_csv("data/reduced_model_fasting.csv", index_col=[0])
    x0 = sips_state_stable.iloc[:,0].values
    x0 = np.hstack((x0[:3], x0[6:9], x0[12:]))

    # Load healthy parameter values (except for max_ba_flow --- disease severity)

    sips_param = np.zeros(27,)

    sips_param[0]  = 0.8236344    # p["synthesis"]
    sips_param[1]  = 0.5300548    # p["syn_frac_CA"]
    sips_param[2]  = 0.1175337    # p["li_to_bd_freq"]
    sips_param[3]  = 0.0025905    # p["bd_to_dsi_freq"]
    sips_param[4]  = 0.0013899    # p["dsi_to_co_freq"]
    sips_param[5]  = 0.0003818    # p["co_to_fe_freq"]
    sips_param[6]  = 3.1500000    # p["pl_volume"]
    sips_param[7]  = 0.8250000    # p["pl_flow_thru_li"]
    sips_param[8]  = 0.9520841    # p["hep_extract_ratio_conj_tri"]
    sips_param[9]  = 0.8383059    # p["hep_extract_ratio_conj_di"]
    sips_param[10] = 0.6250000    # p["hep_extract_ratio_unconj_to_conj"]
    sips_param[11] = 0.0030990    # p["gut_deconj_freq_co"]
    sips_param[12] = 0.2897722    # p["gut_deconj_freq_dsi_to_co"]
    sips_param[13] = 0.0027620    # p["gut_biotr_freq_CA"]
    sips_param[14] = 0.7828888    # p["gut_biotr_freq_CDCA_to_CA"]
    sips_param[15] = 0.0436090    # p["max_asbt_rate"]
    sips_param[16] = 0.0001918    # p["gut_pu_freq_co"]
    sips_param[17] = 0.5986804    # p["gut_pu_freq_dsi_to_co"]
    sips_param[18] = 84.000000    # p["meal_reflex_loc_bd"]
    sips_param[19] = 12.000000    # p["meal_reflex_loc_dsi"]
    sips_param[20] = 10.250000    # p["meal_reflex_loc_co"]
    sips_param[21] = 7.1422496    # p["meal_reflex_peak_bd"]
    sips_param[22] = 4.9142464    # p["meal_reflex_peak_dsi"]
    sips_param[23] = 3.6450010    # p["meal_reflex_peak_co"]
    sips_param[24] = 0.0000145    # p["co_sulfate_freq"]
    sips_param[25] = 10000.000    # p["bd_max_ba"]
    sips_param[26] = max_ba_flow  # p["bd_max_flow"]
    
    PSC_data = pd.read_csv("data/PSC_fasting_for_reduced_model.csv", index_col=[0])
    return x0, sips_param, PSC_data

class TensorboardCallback(BaseCallback):

    def __init__(self, check_freq: int, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:        
            self.logger.record("info/cholesterol_elimination", self.locals["infos"][0]["cholesterol_elimination"])
            self.logger.record("info/toxicity", self.locals["infos"][0]["toxicity"])
            self.logger.record("info/digestion", self.locals["infos"][0]["digestion"])
            self.logger.record("info/fitting_error", self.locals["infos"][0]["fitting_error"])
            self.logger.record("info/param_deviation", self.locals["infos"][0]["param_deviation"])            
        return True