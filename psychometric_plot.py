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
fig, ax = plot_psychometric(trials)





# %%
