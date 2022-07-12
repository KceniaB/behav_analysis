#%% 
""" 
How to git clone this repository? 

Terminal
cd IBL/fromGitHub/ 
git clone https://github.com/KceniaB/behav_analysis.git
KceniaB
enter the token 
DONE 
"""

"""
KceniaBougrova 
12July2022 
Psychometric curve based on: 
https://github.com/int-brain-lab/ibllib/blob/master/examples/loading_data/loading_trials_data.ipynb
""" 
#%% 
from one.api import ONE 
ONE() 
one = ONE() 
import numpy as np
import pandas as pd
from behav_functions import load_trials 
import matplotlib.pyplot as plt

#check the eids from a certain subject
eids = one.search(subject='ZFM-04019') 
len(eids)
eid = eids[12] 
#another example session eid, from the link above in the beginning of this file 
#eid = '4ecb5d24-f5cc-402c-be28-9d0f7cb14b3a' 
df_Trials = load_trials(eid) 
plt.plot(df_Trials.choice)
plt.show()

#%% Load the trials and plot the psychometric curve 
trials = one.load_object(eid, 'trials', collection='alf')
from brainbox.behavior.training import compute_performance
performance, contrasts, n_contrasts = compute_performance(trials)
performance, contrasts, n_contrasts = compute_performance(trials, prob_right=True)
performance, contrasts, n_contrasts = compute_performance(trials, block=0.8)
from brainbox.behavior.training import plot_psychometric
def compute_performance(trials, signed_contrast=None, block=None, prob_right=False):
    """
    Compute performance on all trials at each contrast level from trials object

    :param trials: trials object that must contain contrastLeft, contrastRight and feedbackType
    keys
    :type trials: dict
    returns: float containing performance on easy contrast trials
    """
    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)

    if block is None:
        block_idx = np.full(trials.probabilityLeft.shape, True, dtype=bool)
    else:
        block_idx = trials.probabilityLeft == block

    if not np.any(block_idx):
        return np.nan * np.zeros(3)

    contrasts, n_contrasts = np.unique(signed_contrast[block_idx], return_counts=True)

    if not prob_right:
        correct = trials.feedbackType == 1
        performance = np.vectorize(lambda x: np.mean(correct[(x == signed_contrast) & block_idx]))(contrasts)
    else:
        rightward = trials.choice == -1
        # Calculate the proportion rightward for each contrast type
        performance = np.vectorize(lambda x: np.mean(rightward[(x == signed_contrast) & block_idx]))(contrasts)

    return performance, contrasts, n_contrasts


def compute_n_trials(trials):
    """
    Compute number of trials in trials object

    :param trials: trials object
    :type trials: dict
    returns: int containing number of trials in session
    """
    return trials['choice'].shape[0]

import brainbox.behavior.pyschofit as psy
def compute_psychometric(trials, signed_contrast=None, block=None, plotting=False):
    """
    Compute psychometric fit parameters for trials object

    :param trials: trials object that must contain contrastLeft, contrastRight and probabilityLeft
    :type trials: dict
    :param signed_contrast: array of signed contrasts in percent, where -ve values are on the left
    :type signed_contrast: np.array
    :param block: biased block can be either 0.2 or 0.8
    :type block: float
    :return: array of psychometric fit parameters - bias, threshold, lapse high, lapse low
    """

    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)

    if block is None:
        block_idx = np.full(trials.probabilityLeft.shape, True, dtype=bool)
    else:
        block_idx = trials.probabilityLeft == block

    if not np.any(block_idx):
        return np.nan * np.zeros(4)

    prob_choose_right, contrasts, n_contrasts = compute_performance(trials, signed_contrast=signed_contrast, block=block,
                                                                    prob_right=True)

    if plotting:
        psych, _ = psy.mle_fit_psycho(
            np.vstack([contrasts, n_contrasts, prob_choose_right]),
            P_model='erf_psycho_2gammas',
            parstart=np.array([0., 40., 0.1, 0.1]),
            parmin=np.array([-50., 10., 0., 0.]),
            parmax=np.array([50., 50., 0.2, 0.2]),
            nfits=10)
    else:

        psych, _ = psy.mle_fit_psycho(
            np.vstack([contrasts, n_contrasts, prob_choose_right]),
            P_model='erf_psycho_2gammas',
            parstart=np.array([np.mean(contrasts), 20., 0.05, 0.05]),
            parmin=np.array([np.min(contrasts), 0., 0., 0.]),
            parmax=np.array([np.max(contrasts), 100., 1, 1]))

    return psych
def get_signed_contrast(trials): 
    """
    Compute signed contrast from trials object

    :param trials: trials object that must contain contrastLeft and contrastRight keys
    :type trials: dict
    returns: array of signed contrasts in percent, where -ve values are on the left
    """
    # Replace NaNs with zeros, stack and take the difference
    contrast = np.nan_to_num(np.c_[trials['contrastLeft'], trials['contrastRight']])
    return np.diff(contrast).flatten() * 100

contrasts_2 = [-100. , -25. , 0. , 25. , 100. ]

def plot_psychometric(trials, ax=None, title=None, **kwargs):
    """
    Function to plot pyschometric curve plots a la datajoint webpage
    :param trials:
    :return:
    """
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.rcParams["figure.dpi"] = 300

    signed_contrast = get_signed_contrast(trials)
    contrasts_fit = np.arange(-100, 100)

    prob_right_50, contrasts_50, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.5, prob_right=True)
    pars_50 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.5, plotting=True)
    prob_right_fit_50 = psy.erf_psycho_2gammas(pars_50, contrasts_fit)

    prob_right_20, contrasts_20, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.2, prob_right=True)
    pars_20 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.2, plotting=True)
    prob_right_fit_20 = psy.erf_psycho_2gammas(pars_20, contrasts_fit)

    prob_right_80, contrasts_80, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.8, prob_right=True)
    pars_80 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.8, plotting=True)
    prob_right_fit_80 = psy.erf_psycho_2gammas(pars_80, contrasts_fit)

    cmap = ["#E07C12","#320F42","#008F7C"]

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    # TODO error bars

    fit_50 = ax.plot(contrasts_fit, prob_right_fit_50, color=cmap[1])
    data_50 = ax.scatter(contrasts_50, prob_right_50, color=cmap[1], alpha=0.5)
    fit_20 = ax.plot(contrasts_fit, prob_right_fit_20, color=cmap[0])
    data_20 = ax.scatter(contrasts_20, prob_right_20, color=cmap[0], alpha=0.5)
    fit_80 = ax.plot(contrasts_fit, prob_right_fit_80, color=cmap[2])
    data_80 = ax.scatter(contrasts_80, prob_right_80, color=cmap[2], alpha=0.5)
    ax.legend([fit_50[0], data_50, fit_20[0], data_20, fit_80[0], data_80],
              ['p_left=0.5 fit', 'p_left=0.5 data', 'p_left=0.2 fit', 'p_left=0.2 data', 'p_left=0.8 fit', 'p_left=0.8 data'],
              loc='lower right',
              fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Probability choosing right')
    ax.set_xlabel('Contrasts') 
    plt.xticks(contrasts_2)
    plt.axhline(y=0.5,color = 'gray', linestyle = '--',linewidth=0.25) 
    plt.axvline(x=0.5,color = 'gray', linestyle = '--',linewidth=0.25) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if title:
        ax.set_title(title)

    return fig, ax
fig, ax = plot_psychometric(trials)
#fig.savefig("test_my1stpsychometricplot.png")






# %%
