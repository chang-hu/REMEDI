# REMEDI

REinforcement learning-driven adaptive MEtabolism modeling of primary sclerosing cholangitis DIsease progression

## Description

Primary sclerosing cholangitis (PSC) is a rare, incurable disease wherein altered bile acid metabolism contributes to sustained liver injury. REMEDI captures bile acid dynamics and the body's adaptive response during PSC progression. REMEDI merges a differential equation (DE)-based mechanistic model that describes bile acid metabolism with reinforcement learning (RL) to continuously emulate the body's adaptations to PSC. An objective of adaptation is to maintain homeostasis by regulating enzymes involved in bile acid metabolism. These enzymes correspond to parameters of the DEs. REMEDI leverages RL to approximate adaptations in PSC, treating homeostasis as a reward signal and the adjustment of the DE parameters as the corresponding actions. 

## Usage

- Use the script `REMEDI_model.py` to train an RL model with specified parameters (see instructions below).
- Use the notebook `REMEDI_analysis.ipynb` to analyze the trained RL model (see instructions inside the notebook).

The following command-line arguments can be used to customize the training process with `REMEDI_model.py`:

```sh
mkdir experiments
python REMEDI_model.py [--algorithm ALGORITHM] [--train_steps TRAIN_STEPS] [--learning_rate LEARNING_RATE] [--n_envs N_ENVS] [--adaptation_days ADAPTATION_DAYS] [--data_ID DATA_ID] [--max_ba_flow MAX_BA_FLOW] [--continue_train_suffix CONTINUE_TRAIN_SUFFIX]
```

**Argument Descriptions:**

1. `--algorithm ALGORITHM`: Specifies the RL algorithm for training the model (Default: PPO).
2. `--train_steps TRAIN_STEPS`: Sets the number of training steps (Default: 4000000).
3. `--learning_rate LEARNING_RATE`: Sets the learning rate for the model (Default: 0.002).
4. `--n_envs N_ENVS`: Specifies the number of vectorized training environments to be used (Default: 16).
5. `--adaptation_days ADAPTATION_DAYS`: Sets the number of adaptation days (Default: 240).
6. `--data_ID DATA_ID`: Specifies the patient identifier (Default: median).
7. `--max_ba_flow MAX_BA_FLOW`: Sets the maximum amount of bile acids allowed to pass through the bile duct to the ileum, in proportion to the degree of bile duct obstruction. (Default: 3.0).
8. `--continue_train_suffix CONTINUE_TRAIN_SUFFIX`: Specifies the checkpoint identifier to resume training from a saved model, skip this option to start a new training session (Default: *None*).

Default settings reproduce results presented in REMEDI.

## Files Overview

1. **REMEDI_model.py**: Main script to train an RL agent to emulate the body's adaptations.

2. **REMEDI_analysis.ipynb**: Notebook to visualize and analyze the trained RL model.

3. **src/sb3_BA_ode.py**: Define the bile acid metabolism extended with PSC pathophysiology using a system of DEs.

4. **src/rl_env.py**: Create the RL environment with the bile acid DEs and specify the corresponding step method, intialization, state and action space, reward calculation etc.

6. **src/rl_util.py**: Function for loading bile acid data and helper function for logging.

5. **src/rl_eval.py**: Functions for visualizing RL results.


## Packages

```sh
pip install -r requirements.txt
```