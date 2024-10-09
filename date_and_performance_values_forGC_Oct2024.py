#%%
"""
KceniaBougrova 
08October2024 

Create a df with the mouse_name, session_date and pm (performance values) 
    eid
    Subject (e.g. ZFM-03448)
    unsigned performance
    signed performance by contrast
        -100
        -50
        -25
        -12
        -06
        00
        06
        12
        25
        50
        100 
""" 

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from brainbox.behavior.training import compute_performance 
from brainbox.io.one import SessionLoader
from one.api import ONE #always after the imports 
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

#%%
""" useful""" 
eids = one.search(project='ibl_fibrephotometry') 

# Initialize the DataFrame
new_df = pd.DataFrame() 
excluded_2 = pd.DataFrame()

# Initialize lists to store values
a = []
b = []
c = []
d = [] 
cn100 = [] 
cn50 = []
cn25 = [] 
cn12 = []
cn06 = []
c00 = []
c06 = []
c12 = []
c25 = [] 
c50 = [] 
c100 = [] 
excluded = [] 
error = []

# Predefined contrast values
contrast_values = np.array([-100.,  -50.,   -25., -12.5, -6.25, 0, 6.25, 12.5, 25.,  50.,  100.])

# Loop through the eids
# for eid in eids[0:10]: 
for eid in eids:
    try: 
        # Load trials and session data
        trials = one.load_object(eid, 'trials')
        ref = one.eid2ref(eid)
        subject = ref.subject
        session_date = str(ref.date)

        # Compute performance and contrasts
        performance, contrasts, n_contrasts = compute_performance(trials)
        
        # Initialize contrast performance dictionary for current session
        contrast_perf_dict = dict(zip(contrasts, performance))
        
        # Performance mean for contrasts -100 and 100
        pm_values = [contrast_perf_dict.get(-100, np.nan), contrast_perf_dict.get(100, np.nan)]
        pm = np.nanmean(pm_values)
        
        # Append eid, subject, session_date, and pm to the lists
        a.append(eid)
        b.append(subject)
        c.append(session_date)
        d.append(str(round(pm, 2)))

        # Append contrast-specific performance values
        cn100.append(contrast_perf_dict.get(-100., np.nan))
        cn50.append(contrast_perf_dict.get(-50., np.nan))
        cn25.append(contrast_perf_dict.get(-25., np.nan))
        cn12.append(contrast_perf_dict.get(-12.5, np.nan))
        cn06.append(contrast_perf_dict.get(-6.25, np.nan))
        c00.append(contrast_perf_dict.get(0, np.nan))
        c06.append(contrast_perf_dict.get(6.25, np.nan))
        c12.append(contrast_perf_dict.get(12.5, np.nan))
        c25.append(contrast_perf_dict.get(25., np.nan))
        c50.append(contrast_perf_dict.get(50., np.nan))
        c100.append(contrast_perf_dict.get(100., np.nan)) 
        print(f"DONE eid {eid}")

    except Exception as e: 
        print(f"excluded eid: {eid} due to error: {e}") 
        excluded.append(eid)
        error.append(e)
        pass

# Create df from the lists
new_df["eid"] = a
new_df["subject"] = b
new_df["session_date"] = c
new_df["unsigned_performance"] = d
new_df["cn100"] = cn100
new_df["cn50"] = cn50
new_df["cn25"] = cn25
new_df["cn12"] = cn12
new_df["cn06"] = cn06
new_df["c00"] = c00
new_df["c06"] = c06
new_df["c12"] = c12
new_df["c25"] = c25
new_df["c50"] = c50
new_df["c100"] = c100 

excluded_2["excluded"] = excluded
excluded_2["error"] = error
# Save the performance values and the error eids into csv 
new_df.to_csv('performance_all_photometry_sessions.csv', index=False) 
excluded_2.to_csv('excluded_photometry_sessions.csv', index=False) 


#%%
""" useful to confirm """
# """ EXTRACT THE 2 COLUMNS OF THE INTERVALS FROM TRIALS """
# # Extract feedback_times separately if it's a 2D array (nested)
# if len(trials['intervals'].shape) == 2:
#     trials['intervals_start'] = trials['intervals'][:, 0]
#     trials['intervals_end'] = trials['intervals'][:, 1]
#     del trials['intervals']  # Remove original nested array

# # Convert to DataFrame again
# trials_df = pd.DataFrame(trials)

# # Display the first few rows
# print(trials_df.head())

# #%% 
# """ LEFT IS NEGATIVE """
# """ LEFT IS NEGATIVE """
# df_trials_corr = df_trials[
#     (df_trials['feedbackType'] == 1) & 
#     ((df_trials['contrastLeft'] == 1.0))
# ]
# df_trials_all = df_trials[
#     ((df_trials['contrastLeft'] == 1.0))
# ]
# len(df_trials_corr)/len(df_trials_all)
]
