#%%
"""
KceniaBougrova 
08November2023 

Create a df with the eid, subject, date, number 

M0 - ZFM-02128
BWM - ZFM-02372
M1 - ZFM-03059
M2 - ZFM-03062
M3 - ZFM-03065
M4 - ZFM-03061
D1 - ZFM-03447
D2 - ZFM-03448
D3 - ZFM-03450
D4 - ZFM-04026
D5 - ZFM-04019
D6 - ZFM-04022
S5 - ZFM-04392
N1T - ZFM-04533
N2T - ZFM-04534
S6 - ZFM-05235
S7 - ZFM-05236
S10 - ZFM-05245
S12 - ZFM-05248
A1 - ZFM-05645
N1 - ZFM-06268
N2 - ZFM-06271
N3 - ZFM-06272
N4 - ZFM-06171
N5 - ZFM-06275
A2 - ZFM-06305
A3 - ZFM-06946
A4 - ZFM-06948 
""" 

#%%
from one.api import ONE 
one = ONE(mode="remote") #new way to load the data KB 01092023
# one = ONE() 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from brainbox.behavior.training import compute_performance 

#%%

all_subjects = ['ZFM-03059', 'ZFM-03062', 'ZFM-03065', 'ZFM-03061', 
                'ZFM-03447', 'ZFM-03448', 'ZFM-03450', 
                'ZFM-04026', 'ZFM-04019', 'ZFM-04022', 
                'ZFM-04392', 'ZFM-04533', 'ZFM-04534', 
                'ZFM-05235', 'ZFM-05236', 'ZFM-05245', 'ZFM-05248', 'ZFM-05645', 
                'ZFM-06268', 'ZFM-06271', 'ZFM-06272', 'ZFM-06171', 'ZFM-06275', 
                'ZFM-06305', 'ZFM-06946', 'ZFM-06948'] 


#%% Load sessions 
new_df = pd.DataFrame()
a = []
b = []
c = []
d = []

for subject in all_subjects: 
    eids = one.search(subject=subject)
    for eid in eids: 
        ref = one.eid2ref(eid)
        a.append(str(eid)) 
        b.append(ref.subject) 
        c.append(str(ref.date))
        d.append(str(ref.sequence))

new_df["eid"], new_df["subject"], new_df["date"], new_df["number"] = [a,b,c,d] 

#print the entire df 
print(new_df.to_string())

#save to csv
new_df.to_csv('all_eid_subj_date_seq.csv')

# %%
