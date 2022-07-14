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
13July2022 
Performance plot based on: 
meeting

14July2022
Performance plot for unsigned contrasts 
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


#%% Load the trials and plot the psychometric curve 
trials = one.load_object(eid, 'trials', collection='alf')
from brainbox.behavior.training import compute_performance
performance, contrasts, n_contrasts = compute_performance(trials)
performance, contrasts, n_contrasts = compute_performance(trials, prob_right=True)
performance, contrasts, n_contrasts = compute_performance(trials, block=0.8) 



#%% Plot 
""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" * * * * * * PLOT FUNCTION * * * * * * """
"""""""""""""""""""""""""""""""""""""""""""""""""""""" 
"""see the WARNING! below""" 
def plot_performance(performance,contrasts,n_contrasts): 
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.rcParams["figure.dpi"] = 100

    #colors 
    pm = np.mean(performance_mean_wo0) 
    if pm <= 0.5: 
        c = "#F94144"
    elif 0.5 <= pm and pm < 0.6: 
        c = "#F3722C" 
    elif 0.6 <= pm and pm < 0.7: 
        c = "#F8961E" 
    elif 0.7 <= pm and pm < 0.75: 
        c = "#F9C74F" 
    elif 0.75 <= pm and pm < 0.85: 
        c = "#90BE6D"
    elif 0.85 <= pm and pm < 0.9: 
        c = "#43AA8B"
    elif 0.9 < pm: 
        c = "#577590"
    else: 
        c = "black"
        print("ERROR")
    
    contrasts_2 = [-100. , -25. , 0. , 25. , 100. ]
    plt.plot(contrasts,performance,c=c,alpha=0.25)
    plt.scatter(contrasts,performance,s=n_contrasts,c=c) 
    plt.axhline(y=0.75,color = 'gray', linestyle = '--',linewidth=0.25) 
    plt.axvline(x=0,color = 'gray', linestyle = '--',linewidth=0.25) 
    plt.title(eid)
    ref = one.eid2ref(eid)
    plt.text(50,0.35, ref.subject + '\n' + str(ref.date) + '\n' + str(round(pm,2)),  color='#ADB6C4', fontsize=12)
    plt.ylim(0.3,1.05)
    plt.ylabel('Performance') 
    plt.xlabel('Contrasts') 
    plt.xticks(contrasts_2)
    plt.show() 
    
    plt.savefig("performance_"+ref.subject+str(ref.date)+eid+".png") 
    return plt

#%% Loop and plot different sessions 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" * * * * * * PLOT DIFFERENT SESSIONS * * * * * * """ 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

for eid in eids[1:len(eids)-1]: 
#for eid in eids[1:3]: 
    try: 
        trials = one.load_object(eid, 'trials', collection='alf')
        ref = one.eid2ref(eid)
        performance, contrasts, n_contrasts = compute_performance(trials)
        performance_mean_wo0 = performance[0:4]
        performance_mean_wo0 = np.append(performance_mean_wo0,performance[5:9])
        fig = plot_performance(performance,contrasts,n_contrasts)
        #fig.savefig("psychometric_"+ref.subject+str(ref.date)+eid+".png") 
    except: 
        pass

#plot_performance(performance,contrasts,n_contrasts) 





#%% Plot unsigned  
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" * * * * * PLOT UNSIGNED CONTRASTS * * * * * """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
WARNING! 
"performance" has all the signed contrasts, but only 1 value for 0, so 0 ends up already calculating the performance mean for -0 and 0
example from one session: 
    performance =   [0.95238095,    0.7804878,  0.55172414,     0.2962963,  0.5375,     0.859375,   0.91176471,     0.88607595,     0.98611111] 
    contrasts =     [-100.  ,       -25.  ,     -12.5 ,         -6.25,      0.  ,       6.25,       12.5 ,          25.  ,          100.] 
    n_contrasts =   [84,            82,         58,             81,         160,        64,         68,             79,             72]
So I can't calculate np.mean(performance) and np.mean(unsigned_performance) directly! It will give me different values! 
Solutions: 
    1. duplicate the performance value from contrast=0 in the mean calculation 
    2. remove the performance value from contrasts=0 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< chosen! 
Added in the previous plot and the next one (performance_mean_wo0)
"""
def unsigning(array): 
    #tested and confirmed
    #input = performance array
    #output = new_array with the means of the same contrasts but with different signs 
    array = performance
    a=[]
    b=[]
    c=[]
    d=[]
    if len(array) == 9: 
        a = np.mean((array[3],array[5]), dtype=float)
        b = np.mean((array[2],array[6]), dtype=float)
        c = np.mean((array[1],array[7]), dtype=float)
        d = np.mean((array[0],array[8]), dtype=float) 
    else: 
        print("ERROR, SESSION DOESN'T HAVE ALL THE CONTRASTS")
    new_array=[]
    new_array = [array[4],a,b,c,d]
    return new_array 

#no need for function 
unsigned_contrasts = [0.,6.25,12.5,25,100]

def unsigning_nc(array): 
    array = n_contrasts
    a=[]
    b=[]
    c=[]
    d=[]
    if len(array) == 9: 
        a = np.sum((array[3],array[5]), dtype=float)
        b = np.sum((array[2],array[6]), dtype=float)
        c = np.sum((array[1],array[7]), dtype=float)
        d = np.sum((array[0],array[8]), dtype=float) 
    new_array=[]for eid in eids[1:len(eids)-1]: 

    new_array = [array[4],a,b,c,d]
    return new_array 

# %% Plot
def plot_performance_2(performance,contrasts,n_contrasts): 
    plt.rcParams['figure.figsize'] = [2.5, 5]
    plt.rcParams["figure.dpi"] = 100

    #colors 
    pm = np.mean(performance_mean_wo0) 
    if pm <= 0.5: 
        c = "#F94144"
    elif 0.5 <= pm and pm < 0.6: 
        c = "#F3722C" 
    elif 0.6 <= pm and pm < 0.7: 
        c = "#F8961E" 
    elif 0.7 <= pm and pm < 0.75: 
        c = "#F9C74F" 
    elif 0.75 <= pm and pm < 0.85: 
        c = "#90BE6D"
    elif 0.85 <= pm and pm < 0.9: 
        c = "#43AA8B"
    elif 0.9 < pm: 
        c = "#577590"
    else: 
        c = "black"
        print("ERROR")
    

    contrasts_2 = [0. , 25. , 100. ]
    plt.plot(contrasts,performance,c=c,alpha=0.25)
    plt.scatter(contrasts,performance,s=n_contrasts,c=c) 
    plt.axhline(y=0.75,color = 'gray', linestyle = '--',linewidth=0.25) 
    #plt.axvline(x=0,color = 'gray', linestyle = '--',linewidth=0.25) 
    plt.title(eid)
    plt.ylim(0.45,1.05)
    ref = one.eid2ref(eid)
    plt.text(50,0.5, ref.subject + '\n' + str(ref.date) + '\n' + str(round(pm,2)),  color='#ADB6C4', fontsize=12)
    plt.ylabel('Performance')
    plt.xlabel('Contrasts') 
    plt.xticks(contrasts_2)
    plt.show() 
    
    plt.savefig("performance_uns"+ref.subject+str(ref.date)+eid+".png") 
    return plt

# %%
for eid in eids[1:len(eids)-1]: 
    try: 
        trials = one.load_object(eid, 'trials', collection='alf')
        ref = one.eid2ref(eid)
        performance, contrasts, n_contrasts = compute_performance(trials)
        unsigned_performance = unsigning(performance) 
        unsigned_n_contrasts = unsigning_nc(n_contrasts)
        performance_mean_wo0 = performance[0:4]
        performance_mean_wo0 = np.append(performance_mean_wo0,performance[5:9])
        fig = plot_performance_2(unsigned_performance,unsigned_contrasts,unsigned_n_contrasts)
    except: 
        pass






# %%
