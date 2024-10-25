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
for eid in eids[665:700]: 
# for eid in eids:
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
# # Save the performance values and the error eids into csv 
# new_df.to_csv('performance_all_photometry_sessions.csv', index=False) 
# excluded_2.to_csv('excluded_photometry_sessions.csv', index=False) 


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
# df_trials = trials_df
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




#%% #########################################################################################################
""" HOW TO SOLVE THE PROBABILITYLEFT BLOCKS BUG """ 
# https://int-brain-lab.github.io/ONE/notebooks/one_list/one_list.html

dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
# values gives the block length 
# example for eid = 'be3208c9-43de-44dc-bdc6-ff8963464f98'
# [90, 27, 82, 50, 30, 30, 31, 78, 64, 83, 24, 42, 74, 72, 34, 41, 52, 56, 68, 39, 45, 88, 37, 35, 29, 69, 85, 52, 37, 78, 80, 28, 68, 95, 34, 36, 42] 

# %%
values_sum = np.cumsum(values) 

#%%
# Initialize a new column 'probL' with NaN values
df_trials['probL'] = np.nan

# Set the first block (first `values_sum[0]` rows) to 0.5
df_trials.loc[:values_sum[0]-1, 'probL'] = 0.5 


df_trials.loc[values_sum[0]:values_sum[1]-1, 'probL'] = df_trials.loc[values_sum[0], 'probabilityLeft']

previous_value = df_trials.loc[values_sum[1]-1, 'probabilityLeft'] 


# Iterate over the blocks starting from values_sum[1]
for i in range(1, len(values_sum)-1):
    print("i = ", i)
    start_idx = values_sum[i]
    end_idx = values_sum[i+1]-1
    print("start and end _idx = ", start_idx, end_idx)
    
    # Assign the block value based on the previous one
    if previous_value == 0.2:
        current_value = 0.8
    else:
        current_value = 0.2
    print("current value = ", current_value)


    # Set the 'probL' values for the current block
    df_trials.loc[start_idx:end_idx, 'probL'] = current_value
    
    # Update the previous_value for the next block
    previous_value = current_value

# Handle any remaining rows after the last value_sum block
if len(df_trials) > values_sum[-1]:
    df_trials.loc[values_sum[-1] + 1:, 'probL'] = previous_value

plt.plot(df_trials.probabilityLeft, alpha=0.5)
plt.plot(df_trials.probL, alpha=0.5)
plt.show() 
# %% #########################################################################################################
""" video data """

test = one.load_object(eid, 'leftCamera', attribute=['lightningPose', 'times'])
video_data = pd.DataFrame(test['lightningPose']) 
video_data["times"] = test.times


# %% #########################################################################################################
""" wheel data """
wheel = one.load_object(eid, 'wheel', collection='alf') 

try:
    # Warning: Some older sessions may not have a wheelMoves dataset
    wheel_moves = one.load_object(eid, 'wheelMoves', collection='alf')
except AssertionError:
    wheel_moves = extract_wheel_moves(wheel.timestamps, wheel.position) 

# https://int-brain-lab.github.io/iblenv/notebooks_external/docs_wheel_moves.html 







































# %% #########################################################################################################
""" loop all through all """ 
#================================================
# CONTRAST LEFT ARE NEGATIVE VALUES IN THE NEW COLUMN allContrasts 
def all_contrasts(df_alldata): 
    df_alldata_2 = df_alldata.reset_index(drop=True)
    array1 = np.array(df_alldata_2["contrastLeft"])
    array3 = np.array(df_alldata_2["contrastRight"]) 
    df_alldata_2["allContrasts"] = 100
    for i in range(0,len(array1)): 
        if array1[i] == 0. or array1[i] == 0.0625 or array1[i] == 0.125 or array1[i] == 0.25 or array1[i] == 0.5 or array1[i] == 1.0: #edited with 0.5 included 20230814
            df_alldata_2["allContrasts"][i] = array1[i] * (-1)
        else: 
            df_alldata_2["allContrasts"][i] = array3[i]
    return(df_alldata_2) 


def plot_check_behav(df_alldata): 
    b = df_alldata
    fig, axs = plt.subplots(3, 2,figsize=(16,10)) 

    axs[0, 0].plot(b.intervals_end-b.intervals_start, alpha=1, color = '#936fac',linewidth=1) 
    axs[0, 0].axhline(y=(b.intervals_end-b.intervals_start).mean(), color = '#936fac', alpha=0.3)
    axs[0, 0].set_title('Trial time (s)')
    axs[0, 0].set(xlabel="intervals_1 - intervals_0") 

    axs[0, 1].plot(b.response_times-b.stimOn_times, alpha=1, color = '#936fac',linewidth=1) 
    axs[0, 1].axhline(y=(b.response_times-b.stimOn_times).mean(), color = '#936fac', alpha=0.3)
    axs[0, 1].set_title('Response time per trial (s)')
    axs[0, 1].set(xlabel="response_times - stimOn_times") 

    axs[1, 0].plot(b.firstMovement_times-b.stimOn_times, alpha=1, color = '#f29222',linewidth=1) 
    axs[1, 0].axhline(y=(b.firstMovement_times-b.stimOn_times).mean(), color = '#f29222', alpha=0.3)
    axs[1, 0].set_title('Reaction time per trial (s)')
    axs[1, 0].set(xlabel="firstMovement_times - stimOn_times")

    axs[1, 1].plot(b.response_times-b.firstMovement_times, alpha=1, color = '#e95e50',linewidth=1) 
    axs[1, 1].axhline(y=(b.response_times-b.firstMovement_times).mean(), color = '#e95e50', alpha=0.3)
    axs[1, 1].set_title('Time since 1st mov per trial (s)')
    axs[1, 1].set(xlabel="response_times - firstMovement_times") 

    axs[2, 0].plot(b.quiescencePeriod, alpha=1, color='#A0C4FF')
    axs[2, 0].set_title('Quiescence period (s)') 
    axs[2, 0].set(xlabel="quiescence") 

    axs[2, 1].plot(b.choice,alpha=0.5, color='#9BF6FF')
    axs[2, 1].plot(b.probabilityLeft, alpha=0.5)
    axs[2, 1].set_title('Choice and ProbabilityLeft')
    axs[2, 1].set(xlabel="choice and probability left") 
    fig.tight_layout()
    fig.show()
    plt.show()

# creating reaction and response time variables 
    #reaction = first mov after stim onset 
def new_time_vars(df_alldata,new_var="test",second_action="firstMovement_times",first_action = "stimOn_times"): 
    df = df_alldata
    df[new_var] = df[second_action] - df[first_action] 
    return df 

# splitting the new_time_vars into correct and incorrect in the df and plotting the histogram/density 
def new_time_vars_c_inc(df_alldata,new_var="reactionTime"): 
    new_var_c = str(new_var+"_c") 
    new_var_inc = str(new_var+"_inc")
    df_alldata[new_var_c] = np.nan
    df_alldata[new_var_inc] = np.nan
    for i in range(0,len(df_alldata[new_var])): 
        if df_alldata["feedbackType"][i] == 1: 
            df_alldata[new_var_c][i] = (df_alldata[new_var][i])
        else: 
            df_alldata[new_var_inc][i] = (df_alldata[new_var][i]) 
    print(new_var," mean time correct = ", np.mean(df_alldata[new_var_c]), " | mean time incorrect = ", np.mean(df_alldata[new_var_inc]))
    return df_alldata 

""" useful""" 
eids = one.search(project='ibl_fibrephotometry') 

excluded = [] 
error = [] 

# Loop through the eids
# for eid in eids[665:700]: 
for eid in eids: 
    try: 
        # Load trials and session data
        trials = one.load_object(eid, 'trials')
        ref = one.eid2ref(eid)
        subject = ref.subject
        session_date = str(ref.date)

        if len(trials['intervals'].shape) == 2:
            trials['intervals_start'] = trials['intervals'][:, 0]
            trials['intervals_end'] = trials['intervals'][:, 1]
            del trials['intervals']  # Remove original nested array

        # # Convert to DataFrame again
        trials_df = pd.DataFrame(trials)
        df_trials = trials_df 

        try: 
            dataset_task_settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json')  
            values = dataset_task_settings.get('LEN_BLOCKS', 'Key not found') 
            # values gives the block length 
            # example for eid = 'be3208c9-43de-44dc-bdc6-ff8963464f98'
            # [90, 27, 82, 50, 30, 30, 31, 78, 64, 83, 24, 42, 74, 72, 34, 41, 52, 56, 68, 39, 45, 88, 37, 35, 29, 69, 85, 52, 37, 78, 80, 28, 68, 95, 34, 36, 42] 

            values_sum = np.cumsum(values) 

            # Initialize a new column 'probL' with NaN values
            df_trials['probL'] = np.nan

            # Set the first block (first `values_sum[0]` rows) to 0.5
            df_trials.loc[:values_sum[0]-1, 'probL'] = 0.5 


            df_trials.loc[values_sum[0]:values_sum[1]-1, 'probL'] = df_trials.loc[values_sum[0], 'probabilityLeft']

            previous_value = df_trials.loc[values_sum[1]-1, 'probabilityLeft'] 


            # Iterate over the blocks starting from values_sum[1]
            for i in range(1, len(values_sum)-1):
                print("i = ", i)
                start_idx = values_sum[i]
                end_idx = values_sum[i+1]-1
                print("start and end _idx = ", start_idx, end_idx)
                
                # Assign the block value based on the previous one
                if previous_value == 0.2:
                    current_value = 0.8
                else:
                    current_value = 0.2
                print("current value = ", current_value)


                # Set the 'probL' values for the current block
                df_trials.loc[start_idx:end_idx, 'probL'] = current_value
                
                # Update the previous_value for the next block
                previous_value = current_value

            # Handle any remaining rows after the last value_sum block
            if len(df_trials) > values_sum[-1]:
                df_trials.loc[values_sum[-1] + 1:, 'probL'] = previous_value

            # plt.plot(df_trials.probabilityLeft, alpha=0.5)
            # plt.plot(df_trials.probL, alpha=0.5)
            # plt.title(f'behavior_{subject}_{session_date}_{eid}')
            # plt.show() 
        except: 
            pass 

        df_alldata = df_trials
        # plot_check_behav(df_alldata) 
        # plt.show()

        freq = df_alldata["feedbackType"].value_counts(normalize=True)*100
        # plt.rcParams["figure.figsize"] = (3,6)
        # freq.to_frame().T.plot.bar(stacked=True, color=["#007e5d","#8c0008"],width=0.1)
        # a = (len(df_alldata.feedbackType[df_alldata.feedbackType==1])/len(df_alldata.feedbackType)*100)
        # plt.title("Performance = "+str((round(a,2)))+"%")
        # plt.gca().set_xticklabels([])
        # plt.title(f'behavior_{subject}_{session_date}_{eid}')
        # plt.show()

        #* contrasts column (negative values are when the stim appeared in the Left side) 
        df_alldata = all_contrasts(df_alldata) 
        #================================================ 
        #* creating stim_Right variable for each side: stim_left is -1 and stim_right is 1 
        #the contrast side is the df_alldata.position 
        #positive is contrastRight 
        # plot_check_behav(df_alldata) 
        # plt.show()

        #================================================
        # creating reaction and response time variables 
            #reaction = first mov after stim onset 
        df_alldata = new_time_vars(df_alldata,new_var="reactionTime",second_action="firstMovement_times",first_action = "stimOn_times")
            #response = time for the decision/wheel movement, from stim on until choice 
        df_alldata = new_time_vars(df_alldata,new_var="responseTime",second_action="response_times",first_action = "stimOn_times")
            #response_mov = time for the decision/wheel movement, from the 1st mov on until choice 
        df_alldata = new_time_vars(df_alldata,new_var="responseTime_mov",second_action="response_times",first_action = "firstMovement_times")
        df_alldata = new_time_vars(df_alldata, new_var="trialTime",second_action="intervals_end",first_action="intervals_start") 

        # # Save the performance values and the error eids into csv 
        df_alldata.to_csv(f'behav_tables/behavior_{subject}_{session_date}_{eid}.csv', index=False) 
        # excluded_2.to_csv('excluded_photometry_sessions.csv', index=False) 





        print(f"DONE eid {subject} {session_date} {eid}")

    except Exception as e: 
        print(f"excluded eid: {eid} due to error: {e}") 
        excluded.append(eid)
        error.append(e)
        pass



# # %% #########################################################################################################
# """ video data """

# test = one.load_object(eid, 'leftCamera', attribute=['lightningPose', 'times'])
# video_data = pd.DataFrame(test['lightningPose']) 
# video_data["times"] = test.times


# # %% #########################################################################################################
# """ wheel data """
# wheel = one.load_object(eid, 'wheel', collection='alf') 

# try:
#     # Warning: Some older sessions may not have a wheelMoves dataset
#     wheel_moves = one.load_object(eid, 'wheelMoves', collection='alf')
# except AssertionError:
#     wheel_moves = extract_wheel_moves(wheel.timestamps, wheel.position) 

# # https://int-brain-lab.github.io/iblenv/notebooks_external/docs_wheel_moves.html 

