#%% 
#===========================================================================
#  ?                                ABOUT
#  @author         :  Kcenia Bougrova
#  @repo           :  KceniaB
#  @createdOn      :  photometry_processing_new 05102022
#  @description    :  process the photometry data and align to the behavior 
#  @lastUpdate     :  2023-07-10
#===========================================================================
#%%
#===========================================================================
#                            1. FILE PATHS
#===========================================================================

#-------------------------------------------- 2023-08-24 ---------------------------------------------  
mouse           = 'A2' #NICE SIGNAL =D
date            = '2023-08-24' 
region          = 'Region4G' 
main_path       = '/home/kcenia/Documents/Photometry_results/' + date + '/' 
session_path    = main_path+'raw_photometry1.csv' 
session_path_behav = main_path + mouse + '/'
io_path         = main_path+'bonsai_DI01.csv' 
init_idx = 800

mouse           = 'N1' #NICE SIGNAL =D
date            = '2023-08-24' 
region          = 'Region3G' 
main_path       = '/home/kcenia/Documents/Photometry_results/' + date + '/' 
session_path    = main_path+'raw_photometry2.csv' 
session_path_behav = main_path + mouse + '/'
io_path         = main_path+'bonsai_DI02.csv' 
init_idx = 500
end_idx = len(df_PhotometryData)-1000 



#%%
#===========================================================================
#                            2. IMPORTS
#===========================================================================
from photometry_processing_new_functions import * 


#%%
df_PhotometryData = pd.read_csv(session_path) 
def new_func(io_path):
    df_raw_phdata_DI0_true = import_DI(io_path)
    return df_raw_phdata_DI0_true

df_raw_phdata_DI0_true = new_func(io_path) 

#%%
#===========================================================================
#                           4.3 BEHAVIOR Bpod data
#===========================================================================
# * * * * * * * * * * LOAD BPOD DATA * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
#! NOT REALLY WORKING 
behav_path = session_path_behav+"/raw_behavior_data/_iblmic_audioOnsetGoCue.times_mic.npy"
# behav_path=test
if path.exists(behav_path)==True: 
    from ibllib.io.extractors.training_trials import extract_all 
    df_alldata = extract_behav_t(session_path_behav) 
    print("trainingCW")
else: 
    from ibllib.io.extractors.biased_trials import extract_all 
    df_alldata = extract_behav_b(session_path_behav)
    #print("biasedCW") 

#%% 
# from one.api import ONE 
# one = ONE(mode="remote") #new way to load the data KB 01092023
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt 
# from brainbox.behavior.training import compute_performance 

# eids = one.search(subject='ZFM-04534') 
# len(eids)
# eid = eids[3]
# ref = one.eid2ref(eid)
# print(ref)

# trials = one.load_object(eid, 'trials', collection='alf') 
# #trials.keys()
# columns = ['included', 'stimOnTrigger_times', 'goCueTrigger_times', 'goCue_times', 'response_times', 'choice', 'stimOn_times', 'contrastLeft', 'contrastRight', 'feedback_times', 'feedbackType', 'rewardVolume', 'probabilityLeft', 'firstMovement_times', 'intervals']
# test = pd.DataFrame(trials, columns=columns[0:14])

# test["intervals_start"] = trials["intervals"][:,0]
# test["intervals_stop"] = trials["intervals"][:,1]
# #create the dataframe
# df_alldata = pd.DataFrame(test) 

# df_alldata.rename(columns={'intervals_start': 'intervals_0', 'intervals_stop': 'intervals_1'}, inplace=True)


#%% 
#================================================
#* contrasts column (negative values are when the stim appeared in the Left side) 
df_alldata = all_contrasts(df_alldata) 
#================================================ 
#* creating stim_Right variable for each side: stim_left is -1 and stim_right is 1 
#the contrast side is the df_alldata.position 
#positive is contrastRight 
plot_check_behav(df_alldata) 

#================================================
# creating reaction and response time variables 
    #reaction = first mov after stim onset 
df_alldata = new_time_vars(df_alldata,new_var="reactionTime",second_action="firstMovement_times",first_action = "stimOn_times")
    #response = time for the decision/wheel movement, from stim on until choice 
df_alldata = new_time_vars(df_alldata,new_var="responseTime",second_action="response_times",first_action = "stimOn_times")
    #response_mov = time for the decision/wheel movement, from the 1st mov on until choice 
df_alldata = new_time_vars(df_alldata,new_var="responseTime_mov",second_action="response_times",first_action = "firstMovement_times")

plot_reactions_and_responses(df_alldata) 

#================================================df_alldata = all_contrasts(df_alldata) 

df_alldata = new_time_vars_c_inc(df_alldata,new_var="reactionTime") 
df_alldata = new_time_vars_c_inc(df_alldata,new_var="responseTime") 
df_alldata = new_time_vars_c_inc(df_alldata,new_var="responseTime_mov") 

plot_reactions_and_responses_diff(df_alldata)

show_plot(df_alldata)

#================================================
# %%
#===========================================================================
#                   5. INTERPOLATE BPOD AND NPH TIME
#===========================================================================

""" 5.1 Check the same length of the events from BPOD (Behav) & TTL (Nph) """
# load the bpod reward times - it was the TTL that was sent to the nph system
#bpod_sync = np.array(df_alldata["feedback_times"])
bpod_sync = np.array(df_alldata["stimOnTrigger_times"])
#bpod_sync = np.array(df_alldata["goCue_times"]) 

# load the TTL reward times - it was the TTL that was sent to the nph system 

nph_sync = np.array(df_raw_phdata_DI0_true["Timestamp"]) 
#to test if they have the same length 
print(len(bpod_sync),len(nph_sync))
# print(nph_sync[-1]-nph_sync[-2],nph_sync[-2]-nph_sync[-3], nph_sync[-3]-nph_sync[-4]) 
# print(bpod_sync[-1]-bpod_sync[-2],bpod_sync[-2]-bpod_sync[-3], bpod_sync[-3]-bpod_sync[-4]) 

#? WHEN TTL ERRORS: 
#bpod_sync = bpod_sync[0:len(nph_sync)-1]


#%%
""" 5.2 Assert & reVerify the length from BPOD (Behav) & TTL (Nph) """ 
nph_sync = assert_length(nph_sync,bpod_sync) 
assert len(bpod_sync) == len(nph_sync), "sync arrays are of different length" 

verify_length_2(nph_sync,bpod_sync) 
#================================================
#%%
""" 5.3 INTERPOLATE """
# * * * * * * * * * * INTERPOLATE * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
def interpolate_times(nph_sync,bpod_sync): 
    """
    matches the values
    x and y should be the same length 
    x = nph_sync 
    y = bpod_sync
    """
    # df_raw_phdata_DI0_true = import_DI(io_path) 
    # nph_sync = np.array(df_raw_phdata_DI0_true["Timestamp"]) 
    # x=nph_sync
    # y=bpod_sync

    nph_to_bpod_times = interp1d(nph_sync, bpod_sync, fill_value='extrapolate') 
    nph_to_bpod_times
    nph_frame_times = df_PhotometryData['Timestamp']
    frame_times = nph_to_bpod_times(nph_frame_times)   # use interpolation function returned by `interp1d`
    plt.plot(nph_sync, bpod_sync, 'o', nph_frame_times, frame_times, '-')
    plt.show() 
    #! ADD RESIDUAL ANALYSIS
    #assert(np.all(np.abs(np.diff(df_DI0['Value.Value'].values) == 1)))
    return frame_times
df_PhotometryData["bpod_frame_times_feedback_times"] = interpolate_times(nph_sync,bpod_sync)
df_470 = df_PhotometryData[df_PhotometryData.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_PhotometryData[df_PhotometryData.LedState==1] 
df_415 = df_415.reset_index(drop=True)
#================================================ 

#%% 
#===========================================================================
#      4. FUNCTIONS TO LOAD DATA AND ADD SOME VARIABLES (BEHAVIOR)
#===========================================================================

df_PhotometryData = LedState_or_Flags(df_PhotometryData)

""" 4.1.2 Check for LedState/previous Flags bugs """ 
""" 4.1.2.1 Length """
# Verify the length of the data of the 2 different LEDs
df_470, df_415 = verify_length(df_PhotometryData)
""" 4.1.2.2 Verify if there are repeated flags """ 
verify_repetitions(df_PhotometryData["LedState"])
""" 4.1.3 Remove "weird" data (flag swap, huge signal) """ 
session_day=date
plot_outliers(df_470,df_415,region,mouse,session_day) 


#%%
""" 7.3 Removing extra TTLs (which are there due to the photometry signal removal during the processing) """
xcoords = nph_sync
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue')
plt.rcParams.update({'font.size': 22})
plt.plot(df_470['Timestamp'],df_470['Region3G'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
plt.show() 

#%% 
""" remove weird signal / cut session """ 
# if 'init_idx' in locals(): 
#     print("idx pre-defined")
#     init_idx = init_idx
# else: 
#     init_idx = 1000 #1000
#     end_idx = len(df_PhotometryData)
#     #end_idx = 300000 #180000
#     print('Variable does not exist') 
init_idx = 100
end_idx = len(df_PhotometryData)

df_PhotometryData_1 = df_PhotometryData[init_idx:end_idx] 
df_PhotometryData_1 = df_PhotometryData_1.reset_index(drop=True)
if (df_PhotometryData_1.LedState[0] == 1) and (df_PhotometryData_1.LedState[len(df_PhotometryData_1.LedState)-1]==1): 
    df_PhotometryData_1 = df_PhotometryData_1[1:len(df_PhotometryData_1)]
else: 
    df_PhotometryData_1 = df_PhotometryData_1
df_PhotometryData_1 = df_PhotometryData_1.reset_index(drop=True)
if (df_PhotometryData_1.LedState[0] == 2) and (df_PhotometryData_1.LedState[len(df_PhotometryData_1.LedState)-1]==2): 
    df_PhotometryData_1 = df_PhotometryData_1[0:len(df_PhotometryData_1)-2] 
df_PhotometryData_1 = df_PhotometryData_1.reset_index(drop=True)

#470nm 
df_470 = df_PhotometryData_1[df_PhotometryData_1.LedState==2] 
# df_470 = df_470.reset_index(drop=True)
#415nm 
df_415 = df_PhotometryData_1[df_PhotometryData_1.LedState==1] 
# df_415 = df_415.reset_index(drop=True)
print("470 = ",df_470.LedState.count()," 415 = ",df_415.LedState.count())
assert len(df_470) == len(df_415), "sync arrays are of different length" 

plt.rcParams["figure.figsize"] = (8,5)  
plt.plot(df_470[region],c='#279F95',linewidth=0.5)
plt.plot(df_415[region],c='#803896',linewidth=0.5) 
plt.title("Cropped signal, what to use next")
plt.legend(["GCaMP","isosbestic"],frameon=False)
sns.despine(left = False, bottom = False) 
plt.show()
#%%
# plot the entire raw signal 
df_470_a = df_PhotometryData[df_PhotometryData.LedState==2] 
df_415_a = df_PhotometryData[df_PhotometryData.LedState==1] 
plt.rcParams["figure.figsize"] = (8,5)  
plt.plot(df_470_a["Timestamp"], df_470_a[region],c='#279F95',linewidth=0.5, alpha=0.5)
plt.plot(df_415_a["Timestamp"], df_470_a[region],c='#803896',linewidth=0.5, alpha=0.5) 
plt.plot(df_470["Timestamp"], df_470[region],c='#279F95',linewidth=0.5)
plt.plot(df_415["Timestamp"], df_415[region],c='#803896',linewidth=0.5) 
xcoords = nph_sync
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue',linewidth=0.1)
plt.title("Entire signal, raw data")
plt.legend(["GCaMP","isosbestic"],frameon=False)
sns.despine(left = False, bottom = False) 
# plt.axvline(x=init_idx) 
# plt.axvline(x=end_idx) 
plt.show()

# %% 
df_PhotometryData = df_PhotometryData_1.reset_index(drop=True)  
df_470 = df_PhotometryData[df_PhotometryData.LedState==2] 
df_470 = df_470.reset_index(drop=True)
df_415 = df_PhotometryData[df_PhotometryData.LedState==1] 
df_415 = df_415.reset_index(drop=True) 
#================================================
""" 4.1.4 FRAME RATE """ 
acq_FR = find_FR(df_470["Timestamp"]) 

#================================================
#%% 
#===========================================================================
#                       6. PHOTOMETRY SIGNAL PROCESSING
#===========================================================================
#===========================================================================
# *                            INFO HEADER
#   I should have: 
#       GCaMP, 
#       isosbestic, 
#       times in nph, 
#       times in bpod, 
#       ttl in nph, 
#       ttl in bpod  
#===========================================================================

raw_reference = df_415[region] #isosbestic 
raw_signal = df_470[region] #GCaMP signal 
raw_timestamps_bpod = df_470["bpod_frame_times_feedback_times"]
raw_timestamps_nph_470 = df_470["Timestamp"]
raw_timestamps_nph_415 = df_415["Timestamp"]
raw_TTL_bpod = bpod_sync
raw_TTL_nph = nph_sync

plt.plot(raw_signal[:],color="#60d394")
plt.plot(raw_reference[:],color="#c174f2") 
plt.legend(["signal","isosbestic"],fontsize=15, loc="best")
plt.show() 
#================================================


#%%
plt.rcParams["figure.figsize"] = (12,8)  
fig, ax1 = plt.subplots()

color = 'tab:green'
ax1.set_xlabel('frames')
ax1.set_ylabel('GCaMP', color=color)
ax1.plot(raw_signal, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:purple'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(raw_reference, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.set_dpi(100)
fig.tight_layout() 
plt.show()
#================================================


#%% 
plt.rcParams["figure.figsize"] = (20,3)  
fig, ax1 = plt.subplots()
color = 'tab:green'
ax1.set_xlabel('frames')
ax1.set_ylabel('GCaMP', color=color)
ax1.plot(raw_signal, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:purple'
ax2.set_ylabel('isosbestic', color=color)  # we already handled the x-label with ax1
ax2.plot(raw_reference, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.set_dpi(100)
fig.tight_layout() 
plt.show() 
#================================================


# %%
#===========================================================================
#      6.1 PHOTOMETRY SIGNAL PROCESSING - according to the GitHub code
#===========================================================================
""" 
1. Smooth
""" 
smooth_win = 10
smooth_reference = smooth_signal(raw_reference, smooth_win)
smooth_signal = smooth_signal(raw_signal, smooth_win) 

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(smooth_signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(smooth_reference,'purple',linewidth=1.5) 
#===========================

#%% 
""" 
2. Find the baseline
""" 
lambd = 5e4 # Adjust lambda to get the best fit
porder = 1
itermax = 50
r_base=airPLS(smooth_reference.T,lambda_=lambd,porder=porder,itermax=itermax)
s_base=airPLS(smooth_signal,lambda_=lambd,porder=porder,itermax=itermax)

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(smooth_signal,'blue',linewidth=1.5)
ax1.plot(s_base,'black',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(smooth_reference,'purple',linewidth=1.5)
ax2.plot(r_base,'black',linewidth=1.5) 
#===========================

#%% 
""" 
3. Remove the baseline and the beginning of the recordings
""" 
remove=500
reference = (smooth_reference[remove:] - r_base[remove:])
signal = (smooth_signal[remove:] - s_base[remove:]) 
timestamps_bpod = raw_timestamps_bpod[remove:]
timestamps_nph_470 = raw_timestamps_nph_470[remove:] 
timestamps_nph_415 = raw_timestamps_nph_415[remove:]
#KB ADDED 

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(reference,'purple',linewidth=1.5) 
#===========================

#%% 
""" 
4. Standardize signals
""" 
z_reference = (reference - np.median(reference)) / np.std(reference)
z_signal = (signal - np.median(signal)) / np.std(signal) 

fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_subplot(211)
ax1.plot(z_signal,'blue',linewidth=1.5)
ax2 = fig.add_subplot(212)
ax2.plot(z_reference,'purple',linewidth=1.5) 
#===========================

#%% 
""" 
5. Fit reference signal to calcium signal using linear regression
""" 
from sklearn.linear_model import Lasso
lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
            positive=True, random_state=9999, selection='random')
n = len(z_reference)
lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
#===========================

timestamps_nph_470 = timestamps_nph_470.reset_index(drop=True)
#%% 
""" 
6. Align reference to signal
""" 
z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,) 

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(z_reference,z_signal,'b.')
ax1.plot(z_reference,z_reference_fitted, 'r--',linewidth=1.5) 

#%%
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(z_signal,'blue')
ax1.plot(z_reference_fitted,'purple') 
#===========================

#%% 
""" 
7. Calculate z-score dF/F 
"""
zdFF = (z_signal - z_reference_fitted)

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.plot(zdFF,'black')

""" """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """ """
#================================================


# %% 
#===========================================================================
#                            7. Joining data
#===========================================================================
""" 7.1 join the timestamp already transformed with the zdFF """
timestamps_nph_470 = timestamps_nph_470.reset_index(drop=True)
timestamps_nph_415 = timestamps_nph_415.reset_index(drop=True)
timestamps_bpod=timestamps_bpod.reset_index(drop=True)
df = pd.DataFrame(timestamps_nph_470)
df = df.rename(columns={'Timestamp': 'timestamps_nph_470'})
df = df.reset_index() 
df["timestamps_nph_415"] = timestamps_nph_415
df["timestamps_bpod"] = timestamps_bpod 
df["zdFF"] = zdFF 
#================================================


#%% 
""" 7.2 adding the raw_reference and the raw_signal """
raw_reference=raw_reference[remove:len(raw_reference)]
raw_signal=raw_signal[remove:len(raw_signal)]
df["raw_reference"] = raw_reference
df["rawsignal"] = raw_signal 
#================================================


#%%
""" 7.3 Removing extra TTLs (which are there due to the photometry signal removal during the processing) """
""" 7.3.1 check the data """ 
df_ttl = pd.DataFrame(raw_TTL_nph, columns=["ttl_nph"])
df_ttl["ttl_bpod"] = raw_TTL_bpod 

xcoords = df_ttl['ttl_bpod']
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue')
plt.rcParams.update({'font.size': 22})
plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
plt.show()
#================================================


#%% 
""" 7.3.2 join the rest of the events to this TTL """
df_events = pd.concat([df_ttl, df_alldata], axis=1)
xcoords = df_events['ttl_bpod']
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue')
plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
plt.show()
#================================================


# %%
""" 7.3.3 remove the TTLs that happen before the photometry signal """ 

valuez_bef = df["timestamps_bpod"][0] #when I start the photometry signal (already cut from the pre-processing)
this_indexz = 0 
for i in range(0,len(df_events)): 
    if df_events.ttl_bpod[i] < valuez_bef: 
        this_valuez = df_events.ttl_bpod[i]
        this_indexz = i
df_events = df_events[int((this_indexz)+1):(len(df_events.ttl_bpod)-1)] 
df_events = df_events.reset_index(drop=True) 

xcoords = df_events['ttl_bpod']
for xc in zip(xcoords):
    plt.axvline(x=xc, color='blue')
#plt.xlim([1275,1300])
plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
#plt.xlim(1900, 1935)
plt.show() 
#================================================ 


#%% remove the TTLs that happen after the photometry signal 
""" RECHECK FOR THE FREEZING EVENTS 
#! WARNING %%%%%%% WRONG """
# valuez_aft = df["timestamps_bpod"][-1:]
# valuez_aft = valuez_aft.reset_index()
# valuez_aft = valuez_aft.timestamps_bpod[0] 
# for i in range(0,len(df_events)): 
#     if df_events.ttl_bpod[i] < valuez_aft: 
#         this_valuez = df_events.ttl_bpod[i]
#         this_indexz = i 
# df_events = df_events[0:int(this_indexz)] #gives an error Nov012021 #solved! -5669 above
# #df_events = df_events.reset_index(drop=True)

# xcoords = df_events['ttl_bpod']
# for xc in zip(xcoords):
#     plt.axvline(x=xc, color='blue')
# #plt.xlim([0,100])
# plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
# plt.rcParams["figure.figsize"] = (20,10)
# #plt.xlim(1900, 1935)
# plt.show() 
#================================================


#%%
#===========================================================================
#           7.4 Restructure the behavior data & add trial column
#===========================================================================
# #* A. list of time variables table_x
# table_x = ["feedback_times",
#         "response_times", 
#         "goCueTrigger_times",
#         "goCue_times", 
#         "stimOnTrigger_times", 
#         "stimOn_times", 
#         "stimOff_times", 
#         "stimFreeze_times", 
#         "itiIn_times", 
#         "stimOffTrigger_times",
#         "stimFreezeTrigger_times",
#         "errorCueTrigger_times",
#         "intervals_0", 
#         "intervals_1", 
#         "firstMovement_times",
#         "wheel_moves_intervals_start" 
#         #"wheel_moves_intervals_stop", 
#         #"wheel_timestamps", 
#         #"peakVelocity_times",
#         ] 

# #* B. list of non-time variables table_y 
# table_y = pd.concat([df_events["ttl_nph"], 
#                             df_events["ttl_bpod"], 
#                             df_events["feedbackType"],
#                             df_events["contrastLeft"], 
#                             df_events["contrastRight"], 
#                             df_events["allContrasts"], #KB added 12-10-2022 
#                             df_events["probabilityLeft"], 
#                             df_events["choice"], 
#                             #df_events["repNum"], 
#                             df_events["rewardVolume"], 
#                             #df_events["stim_Right"], 
#                             df_events["reactionTime"],
#                             df_events["reactionTime_triggerTime"],
#                             df_events["feedback_correct"], 
#                             df_events["feedback_incorrect"], 
#                             df_events["wheel_moves_peak_amplitude"], #KB added 12-10-2022
#                             df_events["is_final_movement"], #KB added 12-10-2022
#                             df_events["phase"], #KB added 12-10-2022 
#                             df_events["position"], #KB added 12-10-2022 
#                             df_events["quiescence"]], #KB added 12-10-2022 
#                             axis=1) 

#%%
#================================================
#  *                    ALTERNATIVE
#    for older sessions
#================================================ 
#* A. list of time variables table_x
table_x = ["feedback_times",
        "response_times", 
        "goCueTrigger_times",
        "goCue_times", 
        "stimOnTrigger_times", 
        "stimOn_times", 
        #"stimOff_times", 
        #"stimFreeze_times", 
        #"itiIn_times", 
        #"stimOffTrigger_times",
        #"stimFreezeTrigger_times",
        #"errorCueTrigger_times",
        #"intervals_start", 
        #"intervals_stop", 
        "intervals_0", 
        "intervals_1", 
        "firstMovement_times",
        #"wheel_moves_intervals_start" 
        #"wheel_moves_intervals_stop", 
        #"wheel_timestamps", 
        #"peakVelocity_times",
        ] 

#* B. list of non-time variables table_y 
table_y = pd.concat([df_events["ttl_nph"], 
                            df_events["ttl_bpod"], 
                            df_events["feedbackType"],
                            df_events["contrastLeft"], 
                            df_events["contrastRight"], 
                            #df_events["cL"],
                            #df_events["cR"],    
                            df_events["allContrasts"], #KB added 12-10-2022 
                            df_events["probabilityLeft"], 
                            df_events["choice"], 
                            #df_events["repNum"], 
                            df_events["rewardVolume"], 
                            #df_events["stim_Right"], 
                            df_events["reactionTime"]],
                            #df_events["reactionTime_triggerTime"],
                            #df_events["f_c_reactionTime"], 
                            #df_events["f_inc_reactionTime"]], 
                            # df_events["wheel_moves_peak_amplitude"], #KB added 12-10-2022
                            # df_events["is_final_movement"], #KB added 12-10-2022
                            # df_events["phase"], #KB added 12-10-2022 
                            # df_events["position"], #KB added 12-10-2022 
                            # df_events["quiescence"]], #KB added 12-10-2022 
                            axis=1) 

#%%
##### KB ADDED 20230816
#* A. list of time variables table_x
table_x = ["feedback_times",
        "response_times", 
        "goCueTrigger_times",
        "goCue_times", 
        "stimOnTrigger_times", 
        "stimOn_times", 
        #"stimOff_times", 
        #"stimFreeze_times", 
        #"itiIn_times", 
        #"stimOffTrigger_times",
        #"stimFreezeTrigger_times",
        #"errorCueTrigger_times",
        #"intervals_start", 
        #"intervals_stop", 
        "intervals_0", 
        "intervals_1", 
        "firstMovement_times",
        #"wheel_moves_intervals_start" 
        #"wheel_moves_intervals_stop", 
        #"wheel_timestamps", 
        #"peakVelocity_times",
        ] 
#* B. list of non-time variables table_y 
table_y = pd.concat([df_events["ttl_nph"], 
                            df_events["ttl_bpod"], 
                            df_events["feedbackType"],
                            df_events["contrastLeft"], 
                            df_events["contrastRight"], 
                            #df_events["cL"],
                            #df_events["cR"],    
                            df_events["allContrasts"], #KB added 12-10-2022 
                            df_events["probabilityLeft"], 
                            df_events["choice"], 
                            #df_events["repNum"], 
                            df_events["rewardVolume"], 
                            #df_events["stim_Right"], 
                            df_events["reactionTime"],
                            #df_events["reactionTime_triggerTime"],
                            #df_events["f_c_reactionTime"], 
                            #df_events["f_inc_reactionTime"]], 
                            # df_events["wheel_moves_peak_amplitude"], #KB added 12-10-2022
                            # df_events["is_final_movement"], #KB added 12-10-2022
                            df_events["phase"], #KB added 16-08-2023 
                            df_events["position"], #KB added 16-08-2023 
                            df_events["responseTime"], #KB added 16-08-2023 
                            df_events["responseTime_mov"]], #KB added 16-08-2023 
                            axis=1) 
#================================================



#%%
#================================================
#  *                    INFO
#    
""" works :D """
""" 
Goal: 
    a table with all the system/mouse outputs/choices (table_y) organized by the times of the events (column "name", for every table_x event) 

Description: 
    "{0}".format(x) substitutes {0} by x
    onetime_allnontime is data from one name from table_x which is the times-associated table into the entire non-associated to time table

    1st for loop: 
        1. I add the times of every table_x name to the whole table_y 
        2. create a column that has the table_x name that was joined to the table_y
        3. rename that column as "times" 

        4. create a data frame (onetime_allnontime_2) with the same names as one of the previously created (onetime_allnontime)

    2nd for loop:  
        5. append in loop the rest of the time created tables into that df created in 4. onetime_allnontime_2 
        6. reset the index
        7. sort the values by the "times" of the events 
        8. drop the nans, associated to the stimFreeze_times, stimFreezeTrigger_times, errorCueTrigger_times 
        9. reset again the index 
        10. add a "trial" column 
"""
# 
#================================================
onetime_allnontime={} 
for x in table_x: 
    for i in range(0, len(table_x)): 
        onetime_allnontime["{0}".format(x)] = pd.concat([df_events["{0}".format(x)], 
                                table_y], axis=1) #join df_events of each table_x to the entire table_y
        onetime_allnontime["{0}".format(x)]["name"] = "{0}".format(x) #names with "name" the column to which table_x time name it is associated to
        onetime_allnontime["{0}".format(x)] = onetime_allnontime["{0}".format(x)].rename(columns={"{0}".format(x): 'times'}) #renames the new created column with "times"

onetime_allnontime_2=pd.DataFrame(onetime_allnontime["feedback_times"]) #create a df with the data of the previous loop's first time event
for x in table_x[1:len(table_x)]: 
    onetime_allnontime_2 = onetime_allnontime_2.append((onetime_allnontime["{0}".format(x)])) #keep appending to the df the rest of the time events
onetime_allnontime_2 = onetime_allnontime_2.reset_index(drop=True) #reset the index
df_events_sorted = onetime_allnontime_2.sort_values(by=['times']) #sort all the rows by the time of the events
# to check what are the nans: 
#test = df_events_sorted[df_events_sorted['times'].isna()]
#test.name.unique()
df_events_sorted = df_events_sorted.dropna(subset=['times']) #drop the nan rows - may be associated to the stimFreeze_times, stimFreezeTrigger_times, errorCueTrigger_times
df_events_sorted = df_events_sorted.reset_index() 
#================================================


#%% 
# #add column for the trials 
# # df_events_sorted["trial"] = 0 
# # n=0
# # for i in range(0,len(df_events_sorted)): 
# #     if df_events_sorted["name"][i] == "intervals_0": 
# #         n = n+1
# #         df_events_sorted["trial"][i] = n

# #     else: 
# #         df_events_sorted["trial"][i] = n 
# # df_events_sorted 

# ## through a function KB 2022Jun02 
# test_trial = []
# def create_trial(name_column): 
#     n=0
#     for i in range(0,len(df_events_sorted)): 
#         if name_column[i] == "intervals_0": 
#             n = n+1
#             test_trial.append(n)

#         else: 
#             test_trial.append(n)
#     return test_trial 
# def create_trial2(name_column): #KB 20221017
#     n=0
#     for i in range(0,len(df_events_sorted)): 
#         if name_column[i] == "intervals_start": 
#             n = n+1 
#             test_trial.append(n)

#         else: 
#             test_trial.append(n)
#     return test_trial 

# if "intervals_0" in df_events_sorted: 
#     test_trial = create_trial(df_events_sorted["name"]) 
# else: 
#     test_trial = create_trial2(df_events_sorted["name"]) 

# df_events_sorted["trial"] = test_trial 


test_trial = []
n=0
for i in range(0,len(df_events_sorted["name"])): 
    if df_events_sorted["name"][i] == "intervals_0": 
        n = n+1 
        test_trial.append(n)
        print("yes", n)
    else:
        test_trial.append(n)    
        print("no", n)
df_events_sorted["trial"] = test_trial 

#%%
#===========================================================================
#                            8
#===========================================================================

"""
The code below creates a column where 1 is whenever a certain event occurs, in this case it is when a new trial starts
"""

""" 8.1 to check the photometry signal and two events during some trials of the session """
df_events_sorted["new_trial"] = 0 
for i in range(0, len(df_events_sorted["name"])): 
    if df_events_sorted["name"][i] == "intervals_0": 
        df_events_sorted["new_trial"][i] = 1
    elif df_events_sorted["name"][i] == "intervals_start": #KB 20221017
        df_events_sorted["new_trial"][i] = 1


xcoords = df_events_sorted[df_events_sorted['new_trial'] == 1]
xcoords2 = df_events_sorted[df_events_sorted['name'] == "stimOn_times"]
for xc in zip(xcoords["times"]):
    plt.axvline(x=xc, color='blue')
for xc in zip(xcoords2["times"]):
    plt.axvline(x=xc, color='orange')
plt.plot(df['timestamps_bpod'],df['zdFF'], color = 'g') 
plt.rcParams["figure.figsize"] = (20,10)
plt.xlim(0, 150)
plt.show() 

#%%
df_events_sorted.to_csv('/home/kcenia/Documents/Photometry_results/csvs/all_preprocessed_photometry_behav/df_events_sorted_'+mouse+'_'+date+'_'+region+'.csv')

#================================================


#%%
""" JUST 1 EVENT """
""" 
join the df of one event to the photometry data
sort the entire merge of both dataframes
count the epoch based on the event 
switch the positive values of the epoch -1(?) 
remove the rows from the event 
"""
# df_events_stimOn_times = df_events_sorted[df_events_sorted['name'] == "stimOn_times"]

""" df_events_sorted and photometry """
df["times"] = df["timestamps_bpod"]
# df_all = df.append(onetime_allnontime["feedback_times"]) 
df_all = df.append(df_events_sorted) 
df_all = df_all.sort_values(by=['times']) 
df_all = df_all.reset_index(drop=True) 

b = df_all 
#%% save all the data to later on process/align/plot 

df_all.to_csv('/home/kcenia/Documents/Photometry_results/csvs/all_preprocessed_photometry_behav/'+mouse+'_'+date+'_'+region+'.csv')

#================================================


#%%
""" 8.2 change "name" to the end - so it is possible to repeat the previous behavior rows throughout the photometry data rows """
cols = list(b.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('name')) #Remove b from list
b = b[cols+['name']] #Create new dataframe with columns in the order you want
#================================================

#%%
#===========================================================================
#  *                                 INFO
"""
error found on Feb212022 - ffill will not work for the contrastL and contrastR variables
because it will fill everything, while one of them should be NaN while the other one has a value 

Solving this below: 
1. placing the columns b.contrastLeft and b.contrastRight to the end 
2. ffil the rest of the behav columns 
3. loop through the 2 contrast columns by checking if it's a new trial or not and 
repeating the values that appear in the first row of that trial b.trial 
*seems to work* 
"""
# 
#===========================================================================
#! b.trial doesnt work
#! b.new_trial gives 1.0 when we have "intervals_0" 
#! 0 is joined in allContrasts 
#! b.all_contrasts_separated does not work 
#? LEAVING LEFT AS NEGATIVE
""" COPY FROM 8.2 change cL and cR to the end - so it is possible to repeat the previous behavior rows throughout the photometry data rows """
cols = list(b.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('contrastLeft')) #Remove b from list
b = b[cols+['contrastLeft']] #Create new dataframe with columns in the order you want
cols = list(b.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('contrastRight')) #Remove b from list
b = b[cols+['contrastRight']] 
cols = list(b.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('new_trial')) #Remove b from list #KB added 02112022
b = b[cols+['new_trial']] 
# cols = list(b.columns.values) #Make a list of all of the columns in the df #KB commented 02112022
# cols.pop(cols.index('allContrasts')) #Remove b from list
# b = b[cols+['allContrasts']]

b_1 = b.loc[:, 'feedbackType':'trial'] #new_trial doesnt work well when doing this... #KB changed to "contrastLeft" instead of "new_trial" 02112022
b_1 = b_1.fillna(method='ffill')
b.loc[:, 'feedbackType':'trial'] = b_1 
b = b.reset_index(drop=True) #added 15 December 2021 KB

#b["all_contrasts_separated"] = np.NaN #KB changed from new_column to all_contrasts 09102022
#b["all_contrasts_0joined"] = np.NaN #KB added 10102022 for joined -0 and 0 (it was like before)
b['new_trial'] = b['new_trial'].fillna(0) #KB added 02112022 1=change of trial 
b_trial = np.array(b.new_trial) #KB changed from "trial" to "new_trial" 02112022 
b_contrastL = np.array(b.contrastLeft) 
b_contrastR = np.array(b.contrastRight) 

# for i in range(1,len(b)): 
#     if b.trial[i] != b.trial[i-1]: 
#         number_1, number_2 = b["contrastLeft"][i], b["contrastRight"][i]
#     else: 
#         b["contrastLeft"][i],b["contrastRight"][i] = number_1,number_2 
number_1=np.nan #KB added 09082022 
number_2=np.nan
for i in range(1,len(b_trial)): 
    if b_trial[i] == 1.: #KB changed 02112022
        number_1, number_2 = b_contrastL[i], b_contrastR[i]
    else: 
        b_contrastL[i],b_contrastR[i] = number_1,number_2 
b.contrastLeft = b_contrastL
b.contrastRight = b_contrastR 
b.contrastLeft = (-1)*b.contrastLeft #KB changed from Right to Left, to be standardized with the rest of the columns 09102022
b.all_contrasts_0separated= b["allContrasts"].map(str) #KB added 10102022 for separated -0 and 0 in unique()
b.contrastLeft = (-1)*b.contrastLeft #KB changed from Right to Left, to be standardized with the rest of the columns 09102022 

#! not needed 
# testL = b["contrastLeft"].map(str)
# testR = b["contrastRight"].map(str)
# b.all_contrasts_separated.fillna(b["contrastRight"], inplace=True) 
# b.all_contrasts_separated.fillna(b["contrastLeft"], inplace=True) 
# # Import math Library
# import math 
# for i in range(0,len(b.contrastLeft)): 
#     if math.isnan(b.all_contrasts_separated[i]): 
#         b.all_contrasts_separated[i] = b.contrastLeft[i]
#================================================


#%%
""" 18Jan2021 """

b = b[(b["name"]=="feedback_times").values | b["name"].isna().values] 
b = b.reset_index(drop=True) #added 15 December 2021 KB

c = df_events_sorted[df_events_sorted["name"]=="feedback_times"] #feedback_times #goCue_times
b=b.reset_index(drop=True) 
c=c.reset_index(drop=True)
b_test = b 
b_test = b_test.reset_index(drop=True)
b_test["epoch"] = np.nan
b_time = np.array(b_test["times"])
b_something = np.array(b_test["epoch"]) 
e_something = np.array(b_test["epoch"]) 

c = df_events_sorted[df_events_sorted["name"]=="feedback_times"]
c_event = np.array(c["times"]) 

# # for i in range(0,len(b_time)): 
# for j in range(0,len(c_event)): 
#     b_ndx = np.nonzero(b_time==c_event[j])[0]
#     e_ndx = b_ndx + np.arange(-30,61)
#     e_ndx[30:91] = e_ndx[30:91] + 1
#     b_something[e_ndx] = np.arange(-30,61) #times framerate and itll be in seconds
#     e_something[e_ndx] = j


# for i in range(0,len(b_time)): 
if acq_FR == 30: 
    for j in range(0,len(c_event)): 
        b_ndx = np.nonzero(b_time==c_event[j])[0]
        e_ndx = b_ndx + np.arange(-30,61)
        e_ndx[30:91] = e_ndx[30:91] + 1
        b_something[e_ndx] = np.arange(-30,61) #times framerate and itll be in seconds
        e_something[e_ndx] = j
elif acq_FR == 15: 
    for j in range(0,len(c_event)): 
        b_ndx = np.nonzero(b_time==c_event[j])[0]
        e_ndx = b_ndx + np.arange(-15,31)
        e_ndx[15:46] = e_ndx[15:46] + 1
        b_something[e_ndx] = np.arange(-15,31) #times framerate and itll be in seconds
        e_something[e_ndx] = j
elif acq_FR == 60: df_415
else: 
    print(">>>>> WEIRD FR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

b_test["epoch"] = b_something 
b_test["epoch_trial"] = e_something 
b_test["epoch_sec"] = b_test["epoch"]/acq_FR


teste = b_test.dropna(subset=['epoch']) 
teste = teste.reset_index(drop=True)
#================================================


#%%
#%% 
tc_dual_to_plot = teste
#%%
#===========================================================================
#                            FUNCTIONS TO PLOT
# KB EDITED 2022Jun02 from plotting_processed_data.py code
#===========================================================================

def plot_feedback_choice(arg1): 
    # Plot the responses of the 5-HT DRN neurons according to x event
    sns.set(rc={'figure.figsize':(21,17),'figure.dpi':150},style="ticks") 
    sns.despine(offset={'left':10,'right':10,'top':20,'bottom':13},trim="True") 
    
    #!change
    alignment = "goCue"
    """COLORS"""
    a = tc_dual_to_plot[arg1].unique() 
    cleanedList = [x for x in a if str(x) != 'nan']
    number_of_colors = len(cleanedList)
    if number_of_colors < 3: 
        colors = ["#ea6276","#32aac8"] 
        palette = sns.color_palette(colors,number_of_colors) 
    else: 
        if (arg1 == ("contrastLeft")) or (arg1 == ("contrastRight")):  
            palette = sns.color_palette("rocket_r",number_of_colors) 
        elif (arg1 == ("probabilityLeft")) or (arg1 == ("repNum")): 
            palette = sns.color_palette("coolwarm",number_of_colors) 
        elif (arg1 == ("allContrasts")): 
            colors = ["#247ba0","#70c1b3","#b2dbbf","#f3ffbd","#ead2ac","#f3ffbd","#b2dbbf","#70c1b3","#247ba0"]
            palette = sns.color_palette(colors,number_of_colors)
        else: 
            colors = ["#BA98CE", "#FF83B0", "#79BEFF"] 
            palette = sns.color_palette(colors,number_of_colors) 

    """"LINES"""
    #sns.lineplot(x="epoch_sec", y="zdFF",
    #            data=tc_dual_to_plot, color= 'lightslategrey', linewidth = 0.25, alpha = 0.2, units="epoch_trial", estimator=None, hue=arg1,palette=palette)#, hue="feedbackType",palette=palette)
    sns.lineplot(x="epoch_sec", y=("zdFF"),
                data=tc_dual_to_plot, color= 'mediumblue', linewidth = 3, alpha = 0.85, hue=arg1,palette=palette)
    plt.axvline(x=0, color = "black", alpha=0.9, linewidth = 3, linestyle="dashed", label = alignment)
    # plt.axhline(y=0, color = "gray", alpha=0.75, linewidth = 1.5, linestyle="dashed")

    """LABELS"""
    plt.title(mouse+' '+region+' '+session_day+" split by "+arg1, y=1.1, fontsize=45) 
    plt.xlabel("time since " + alignment +" (s)", labelpad=20, fontsize=60)
    plt.ylabel("zdFF", labelpad=20, fontsize=60) 
    plt.ylim(-1,2) 
    #sns.set(rc={'figure.figsize':(21,17)},style="ticks") #KB commented 09-10-2022 used to use this one
    sns.set(rc={'figure.figsize':(8,6)}, style="ticks") #KB added 09-10-2022 
    sns.despine(offset={'left':10,'right':10,'top':20,'bottom':13},trim="True")
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    leg = plt.legend(loc="upper right", fontsize=35,frameon=False)
    # leg = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=25,frameon=False)

    for line in leg.get_lines():
        line.set_linewidth(5.5)


    """AXIS and DIMENSION"""
    #plt.xlim(-30,60)
    plt.ylim(-1,2) 
    #sns.set(rc={'figure.figsize':(15,10)}) 
    #plt.savefig(save_the_figs+arg1+'_feedback.png') #,dpi=1200) 
    plt.savefig('/home/kcenia/Documents/Photometry_results/Plots/Dec2022/'+mouse+'_'+region+'_'+session_day+'_'+alignment+'_'+arg1+'.png', dpi=300, bbox_inches = "tight") 
    plt.show()


#%%
b_1 = tc_dual_to_plot.loc[:, 'feedbackType':'rewardVolume'] 
a=[]
for col in b_1.columns: 
    a.append(str(col))
b_2 = tc_dual_to_plot.loc[:, 'contrastLeft':'contrastRight']
for col in b_2.columns: 
    a.append(str(col))

for i in range(0,len(a)): 
    plot_feedback_choice(a[i]) 
#================================================

#%% 
print(session_path,io_path,session_path_behav) 
#================================================


#%%
tc_dual_to_plot.to_csv('/home/kcenia/Documents/Photometry_results/csvs/'+mouse+'_'+session_day+'_'+region+'_feedback.csv') 
tc_dual_to_plot.to_csv('/home/kcenia/Documents/Photometry_results/csvs/'+mouse+'_'+session_day+'_'+region+'_goCue.csv') 
#tc_dual_to_plot.to_csv('/home/kcenia/Documents/Photometry_results/SfN_02Nov2022/D5_2022-06-10_goCue.csv') 
#================================================