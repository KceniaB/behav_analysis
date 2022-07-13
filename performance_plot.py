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




#%% Plot 
# %% 
def plot_performance(performance,contrasts,n_contrasts): 
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.rcParams["figure.dpi"] = 100

    #colors 
    pm = np.mean(performance) 
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
    plt.text(50,0.35, ref.subject + '\n' + str(ref.date),  color='#ADB6C4', fontsize=12)
    plt.ylim(0.3,1)
    plt.ylabel('Performance')
    plt.xlabel('Contrasts') 
    plt.xticks(contrasts_2)
    plt.show() 
    
    #plt.savefig("performance_"+ref.subject+str(ref.date)+eid+".png") 
    return plt

#%% Loop and plot different sessions 

for eid in eids[1:10]: 
#for eid in eids[1:3]: 
    trials = one.load_object(eid, 'trials', collection='alf')
    ref = one.eid2ref(eid)
    performance, contrasts, n_contrasts = compute_performance(trials)

    fig = plot_performance(performance,contrasts,n_contrasts)
    #fig.savefig("psychometric_"+ref.subject+str(ref.date)+eid+".png") 

#plot_performance(performance,contrasts,n_contrasts) 
#%% Plot unsigned 

def unsigning(array): 
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
    new_array=[]
    new_array = [array[4],a,b,c,d]
    return new_array 
test = unsigning(performance) 
test_2 = [0.,6.25,12.5,25,100]

def unsigning(array): 
    array = performance
    a=[]
    b=[]
    c=[]
    d=[]
    if len(array) == 9: 
        a = np.sum((array[3],array[5]), dtype=float)
        b = np.sum((array[2],array[6]), dtype=float)
        c = np.sum((array[1],array[7]), dtype=float)
        d = np.sum((array[0],array[8]), dtype=float) 
    new_array=[]
    new_array = [array[4],a,b,c,d]
    return new_array 
test_3 = unsigning(n_contrasts)
# %% 
def plot_performance_2(performance,contrasts,n_contrasts): 
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.rcParams["figure.dpi"] = 100

    #colors 
    pm = np.mean(performance) 
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
    #plt.axhline(y=0.75,color = 'gray', linestyle = '--',linewidth=0.25) 
    #plt.axvline(x=0,color = 'gray', linestyle = '--',linewidth=0.25) 
    plt.title(eid)
    ref = one.eid2ref(eid)
    #plt.text(50,0.35, ref.subject + '\n' + str(ref.date),  color='#ADB6C4', fontsize=12)
    #plt.ylim(0.3,1)
    plt.ylabel('Performance')
    plt.xlabel('Contrasts') 
    plt.xticks(contrasts_2)
    plt.show() 
    
    #plt.savefig("performance_"+ref.subject+str(ref.date)+eid+".png") 
    return plt

# %%

    trials = one.load_object(eid, 'trials', collection='alf')
    ref = one.eid2ref(eid)
    performance, contrasts, n_contrasts = compute_performance(trials)

    fig = plot_performance_2(test,test_2,test_3)






# %%
