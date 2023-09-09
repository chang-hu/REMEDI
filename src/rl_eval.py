import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

linestyle_dict = {'o':'-', '+':'--', '-':':'}

names_states = ['li_cCA', 'li_cCDCA', 'li_cSBA', 'li_uCA', 'li_uCDCA', 'li_uSBA',
                'bd_cCA', 'bd_cCDCA', 'bd_cSBA', 'bd_uCA', 'bd_uCDCA', 'bd_uSBA',
                'dsi_cCA', 'dsi_cCDCA', 'dsi_cSBA', 'dsi_uCA', 'dsi_uCDCA', 'dsi_uSBA',
                'co_cCA', 'co_cCDCA', 'co_cSBA', 'co_uCA', 'co_uCDCA', 'co_uSBA',
                'pl_cCA', 'pl_cCDCA', 'pl_cSBA', 'pl_uCA', 'pl_uCDCA', 'pl_uSBA',
                'fe_cCA', 'fe_cCDCA', 'fe_cSBA', 'fe_uCA', 'fe_uCDCA', 'fe_uSBA']

def plot_sol(sol, t, axes, color, label, direction='o'):
    linestyle = linestyle_dict[direction]
    state_idx = 0
    for idx in range(30):
        axes[idx].set_title(f'{names_states[idx]} level [\u03BCmol]')
        if idx not in [3,4,5,9,10,11]:
            axes[idx].plot(t[:], sol[:, state_idx], linestyle=linestyle, color=color, label=label, alpha=0.6)
            state_idx += 1
        else:
            axes[idx].plot(t[:], np.zeros(shape=sol[:, state_idx].shape), linestyle=linestyle, color=color, label=label, alpha=0.6)
        
        if max(t) == 1440:
            axes[idx].set_xticks((1440 / 4) * np.arange(5))
            axes[idx].set_xticklabels(["8AM", "2PM", "8PM", "2AM", "8AM"])
        else:
            xtick_interval = round((len(t)//6)/10)*10
            xticks = [i for i in t[:] if i % (1440*xtick_interval) == 0]
            axes[idx].set_xticks(xticks)
            axes[idx].set_xticklabels([int(i/1440) for i in xticks])

def plot_entire_duration(states, timepoints, N_STATE, ground_truth=None):
    fig, axes = plt.subplots(5, 6, figsize=(24, 16))
    axes = axes.flatten()

    plot_sol(states[:, :N_STATE], timepoints, axes, color='black', label='RL')

    if ground_truth!=None:
        PSC_data = pd.read_csv("data/PSC_fasting_for_reduced_model.csv", index_col=[0])[f"PSC_log10_{ground_truth}"].apply(lambda x: np.power(10, x)).values
        sips_state_stable = pd.read_csv("data/reduced_model_fasting.csv", index_col=[0]).iloc[-6:,0].values
    
        for i, idx in enumerate(np.arange(24,30)):
            axes[idx].axhline(PSC_data[i], color="red", linestyle="--", label=f"PSC {ground_truth}")
            axes[idx].axhline(sips_state_stable[i], color="blue", linestyle="--", label='Healthy')
            axes[idx].set_xlabel("day")
    
    axes[24].legend(ncol=6, bbox_to_anchor=(-0.05, -0.25), loc='upper left')
    for ax in axes:
        ax.grid(which='both', linestyle='--')
    plt.subplots_adjust(hspace=0.25)
    
def plot_one_day(traj):
    fig, axes = plt.subplots(5, 6, figsize=(24, 16))
    axes = axes.flatten()

    plot_sol(traj.T, np.linspace(0, 1440, 1441), axes, color='black', label='RL')

    for ax in axes:
        ax.grid(which='both', linestyle='--')
    plt.subplots_adjust(hspace=0.25)