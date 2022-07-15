#%%
"""
KceniaBougrova 
15July2022 

Create a df with the mouse_name, session_date and pm (performance values) 

Done for: 
M1
M3
M4
D1
D4
D5
D6 
""" 

#%%
from one.api import ONE 
ONE() 
one = ONE() 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from brainbox.behavior.training import compute_performance

#%% Load sessions from a certain animal
eids = one.search(subject='ZFM-03059') 

#%% 
new_df = pd.DataFrame()
a = []
b = []
c = []
d = []

for eid in eids[0:len(eids)-1]: 
#for eid in eids[1:3]: 
    try: 
        trials = one.load_object(eid, 'trials', collection='alf')
        ref = one.eid2ref(eid)
        mouse_name = ref.subject
        session_date = str(ref.date)
        performance, contrasts, n_contrasts = compute_performance(trials)
        performance_mean_wo0 = performance[0:4]
        performance_mean_wo0 = np.append(performance_mean_wo0,performance[5:9]) 
        pm = np.mean(performance_mean_wo0) 

        
        a.append(mouse_name)
        b.append(session_date)
        c.append(str(round(pm,2)))

    except: 
        pass
new_df["mouse_name"], new_df["session_date"], new_df["pm"] = [a,b,c] 

#%%for an example mouse
new_df_M1 = new_df
new_df_M1
#print the entire df 
print(new_df_M1.to_string())

#save to csv
#new_df_M1.to_csv('date_and_performance_values_M1.csv')
