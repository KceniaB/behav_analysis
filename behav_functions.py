#%% 
from one.api import ONE 
ONE() 
one = ONE() 
import numpy as np
import pandas as pd 

#%%
#others: 
#wheel = one.load_object(eids[0],'wheel')
#ref = one.eid2ref(eid)

#%%
""" 
1. load_trials
"""
def load_trials(eid, laser_stimulation=False, invert_choice=False, invert_stimside=False, one=None):
    import numpy as np
    import pandas as pd
    if one is None:
        from one.api import ONE 
        ONE() 
        one = ONE() 

    trials = pd.DataFrame()
    if laser_stimulation:
        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
         trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
         trials['feedbackType'], trials['choice'],
         trials['feedback_times'], trials['firstMovement_times'], trials['laser_stimulation'],
         trials['laser_probability']) = one.load(
                             eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
                                                 'trials.goCue_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice',
                                                 'trials.feedback_times', 'trials.firstMovement_times',
                                                 '_ibl_trials.laser_stimulation',
                                                 '_ibl_trials.laser_probability'])
        if trials.shape[0] == 0:
            return
        if trials.loc[0, 'laser_stimulation'] is None:
            trials = trials.drop(columns=['laser_stimulation'])
        if trials.loc[0, 'laser_probability'] is None:
            trials = trials.drop(columns=['laser_probability'])
    else:
#        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
#          trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
#          trials['feedbackType'], trials['choice'], trials['firstMovement_times'],
#          trials['feedback_times']) = one.load(
#                              eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
#                                                  'trials.goCue_times', 'trials.probabilityLeft',
#                                                  'trials.contrastLeft', 'trials.contrastRight',
#                                                  'trials.feedbackType', 'trials.choice',
#                                                  'trials.firstMovement_times',
#                                                  'trials.feedback_times'])
        try:
            trials = one.load_object(eid, 'trials') #210810 Updated by brandon due to ONE update
        except:
            return {}
            
            
            
    if len(trials['probabilityLeft']) == 0: # 210810 Updated by brandon due to ONE update
        return
#     if trials.shape[0] == 0:
#         return
#     trials['signed_contrast'] = trials['contrastRight']
#     trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
#     trials['correct'] = trials['feedbackType']
#     trials.loc[trials['correct'] == -1, 'correct'] = 0
#     trials['right_choice'] = -trials['choice']
#     trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
#     trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
#     trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
#     trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
#                'stim_side'] = 1
#     trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
#                'stim_side'] = -1
    assert np.all(np.logical_xor(np.isnan(trials['contrastRight']),np.isnan(trials['contrastLeft'])))
    
    trials['signed_contrast'] = np.copy(trials['contrastRight'])
    use_trials = np.isnan(trials['signed_contrast'])
    trials['signed_contrast'][use_trials] = -np.copy(trials['contrastLeft'])[use_trials]
    trials['correct'] = trials['feedbackType']
    use_trials = (trials['correct'] == -1)
    trials['correct'][use_trials] = 0
    trials['right_choice'] = -np.copy(trials['choice'])
    use_trials = (trials['right_choice'] == -1)
    trials['right_choice'][use_trials] = 0
    trials['stim_side'] = (np.isnan(trials['contrastLeft'])).astype(int)
    use_trials = (trials['stim_side'] == 0)
    trials['stim_side'][use_trials] = -1
#     if 'firstMovement_times' in trials.columns.values:
    trials['reaction_times'] = np.copy(trials['firstMovement_times'] - trials['goCue_times'])
    if invert_choice:
        trials['choice'] = -trials['choice']
    if invert_stimside:
        trials['stim_side'] = -trials['stim_side']
        trials['signed_contrast'] = -trials['signed_contrast']
    return trials
# %%
""" 
2. plot_psychometric 
https://github.com/int-brain-lab/ibllib/blob/master/brainbox/behavior/training.py
""" 

def plot_psychometric(trials, ax=None, title=None, **kwargs):
    """
    Function to plot pyschometric curve plots a la datajoint webpage
    :param trials:
    :return:
    """

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

    cmap = sns.diverging_palette(20, 220, n=3, center="dark")

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    # TODO error bars

    fit_50 = ax.plot(contrasts_fit, prob_right_fit_50, color=cmap[1])
    data_50 = ax.scatter(contrasts_50, prob_right_50, color=cmap[1])
    fit_20 = ax.plot(contrasts_fit, prob_right_fit_20, color=cmap[0])
    data_20 = ax.scatter(contrasts_20, prob_right_20, color=cmap[0])
    fit_80 = ax.plot(contrasts_fit, prob_right_fit_80, color=cmap[2])
    data_80 = ax.scatter(contrasts_80, prob_right_80, color=cmap[2])
    ax.legend([fit_50[0], data_50, fit_20[0], data_20, fit_80[0], data_80],
              ['p_left=0.5 fit', 'p_left=0.5 data', 'p_left=0.2 fit', 'p_left=0.2 data', 'p_left=0.8 fit', 'p_left=0.8 data'],
              loc='upper left')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Probability choosing right')
    ax.set_xlabel('Contrasts')
    if title:
        ax.set_title(title)

    return fig, ax












# %%
