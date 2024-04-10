# This python script will preprocess all subjects' raw EEG data and save it as an individual .fif file that will be used for classification.

# Step 1: load the matlab files and create mne raw objects from them
# Step 2: create events array from the 65th channel, drop channel and define event_id dictionaries
# Step 3: set annotations from events for the raw objects
# For each raw object:
# - Step 4: read channel locations from custom montage file and set it to the raw objects
# - Step 5: apply bandpass filter from 1Hz to 45Hz
# - Step 6: plot the data: select bad channels for interpolation
# - Step 7: perform ICA decomposition and plot the components
# - Step 8: from visual inspection, select the components to remove, reconstruct the sensor signals with artifacts removed
# Step 9: concatenate the raw objects together and save the raw object as a .fif file

# import the necessary modules
import mne
import matplotlib.pyplot as plt
import numpy as np
import mat73

# define the path to the EEG data
data_path = "subject_4"

# define the file names
file_names = [
    "s4_t1_down_p1.mat",
    "s4_t1_down_p2.mat",
    "s4_t1_up_p1.mat",
    "s4_t1_up_p2.mat",
    "s4_t2_p1.mat",
    "s4_t2_p2.mat",
    "s4_t2_p3.mat",
    "s4_t2_p4.mat"
]

# load the files and extract the EEG data iteratively
eeg_data = []
for file_name in file_names:
    data = mat73.loadmat(data_path + "/" + file_name)
    eeg_data.append(data['EEG_rec'][1:, :]) # exclude the first row of time stamps

# Access individual EEG data
eeg_data_1 = eeg_data[0]
eeg_data_2 = eeg_data[1]
eeg_data_3 = eeg_data[2]
eeg_data_4 = eeg_data[3]
eeg_data_5 = eeg_data[4]
eeg_data_6 = eeg_data[5]
eeg_data_7 = eeg_data[6]
eeg_data_8 = eeg_data[7]

# create an MNE information object for each EEG data
sampling_rate = 250
info_1 = mne.create_info(ch_names=[f"ch_{i + 1}" for i in range(eeg_data_1.shape[0])], sfreq=sampling_rate, ch_types='eeg')
info_2 = mne.create_info(ch_names=[f"ch_{i + 1}" for i in range(eeg_data_2.shape[0])], sfreq=sampling_rate, ch_types='eeg')
info_3 = mne.create_info(ch_names=[f"ch_{i + 1}" for i in range(eeg_data_3.shape[0])], sfreq=sampling_rate, ch_types='eeg')
info_4 = mne.create_info(ch_names=[f"ch_{i + 1}" for i in range(eeg_data_4.shape[0])], sfreq=sampling_rate, ch_types='eeg')
info_5 = mne.create_info(ch_names=[f"ch_{i + 1}" for i in range(eeg_data_5.shape[0])], sfreq=sampling_rate, ch_types='eeg')
info_6 = mne.create_info(ch_names=[f"ch_{i + 1}" for i in range(eeg_data_6.shape[0])], sfreq=sampling_rate, ch_types='eeg')
info_7 = mne.create_info(ch_names=[f"ch_{i + 1}" for i in range(eeg_data_7.shape[0])], sfreq=sampling_rate, ch_types='eeg')
info_8 = mne.create_info(ch_names=[f"ch_{i + 1}" for i in range(eeg_data_8.shape[0])], sfreq=sampling_rate, ch_types='eeg')

# create MNE Raw objects
raw_1 = mne.io.RawArray(eeg_data_1, info_1)
raw_2 = mne.io.RawArray(eeg_data_2, info_2)
raw_3 = mne.io.RawArray(eeg_data_3, info_3)
raw_4 = mne.io.RawArray(eeg_data_4, info_4)
raw_5 = mne.io.RawArray(eeg_data_5, info_5)
raw_6 = mne.io.RawArray(eeg_data_6, info_6)
raw_7 = mne.io.RawArray(eeg_data_7, info_7)
raw_8 = mne.io.RawArray(eeg_data_8, info_8)

# create events arrays for each raw object
events_1 = mne.find_events(raw_1, stim_channel='ch_65')
events_2 = mne.find_events(raw_2, stim_channel='ch_65')
events_3 = mne.find_events(raw_3, stim_channel='ch_65')
events_4 = mne.find_events(raw_4, stim_channel='ch_65')
events_5 = mne.find_events(raw_5, stim_channel='ch_65')
events_6 = mne.find_events(raw_6, stim_channel='ch_65')
events_7 = mne.find_events(raw_7, stim_channel='ch_65')
events_8 = mne.find_events(raw_8, stim_channel='ch_65')

# remove the event channel
raw_1.drop_channels(['ch_65'])
raw_2.drop_channels(['ch_65'])
raw_3.drop_channels(['ch_65'])
raw_4.drop_channels(['ch_65'])
raw_5.drop_channels(['ch_65'])
raw_6.drop_channels(['ch_65'])
raw_7.drop_channels(['ch_65'])
raw_8.drop_channels(['ch_65'])

# create event_id dictionaries
event_dict_1 = {
    'motor execution down': 1,
    'visual perception up': 2,
    'imagery down': 3,
    'imagery and perception down': 4
}
event_dict_2 = {
    'motor execution up': 1,
    'visual perception down': 2,
    'imagery up': 3,
    'imagery and perception up': 4
}
event_dict_3 = {
    'imagery up': 1,
    'imagery and perception up': 2,
    'imagery down': 3,
    'imagery and perception down': 4
}

# set annotations from events for the first 2 raw objects
mapping_1 = {
    1: 'motor execution down',
    2: 'visual perception up',
    3: 'imagery down',
    4: 'imagery and perception down'
}
# raw 1,2 - t1 down p1/2
annot_from_events_1 = mne.annotations_from_events(
    events=events_1,
    event_desc=mapping_1,
    sfreq=raw_1.info["sfreq"],
    orig_time=raw_1.info["meas_date"],
)
annot_from_events_2 = mne.annotations_from_events(
    events=events_2,
    event_desc=mapping_1,
    sfreq=raw_2.info["sfreq"],
    orig_time=raw_2.info["meas_date"],
)
raw_1.set_annotations(annot_from_events_1)
raw_2.set_annotations(annot_from_events_2)

# set annotations from events for the next 2 raw objects
mapping_2 = {
    1: 'motor execution up',
    2: 'visual perception down',
    3: 'imagery up',
    4: 'imagery and perception up'
}
# raw 3,4 - t1 up p1/2
annot_from_events_3 = mne.annotations_from_events(
    events=events_3,
    event_desc=mapping_2,
    sfreq=raw_3.info["sfreq"],
    orig_time=raw_3.info["meas_date"],
)
annot_from_events_4 = mne.annotations_from_events(
    events=events_4,
    event_desc=mapping_2,
    sfreq=raw_4.info["sfreq"],
    orig_time=raw_4.info["meas_date"],
)
raw_3.set_annotations(annot_from_events_3)
raw_4.set_annotations(annot_from_events_4)

# set annotations from events for the last 4 raw objects
mapping_3 = {
    1: 'imagery up',
    2: 'imagery and perception up',
    3: 'imagery down',
    4: 'imagery and perception down'
}
# raw 5,6,7,8 - t2 p1/2/3/4
annot_from_events_5 = mne.annotations_from_events(
    events=events_5,
    event_desc=mapping_3,
    sfreq=raw_5.info["sfreq"],
    orig_time=raw_5.info["meas_date"],
)
annot_from_events_6 = mne.annotations_from_events(
    events=events_6,
    event_desc=mapping_3,
    sfreq=raw_6.info["sfreq"],
    orig_time=raw_6.info["meas_date"],
)
annot_from_events_7 = mne.annotations_from_events(
    events=events_7,
    event_desc=mapping_3,
    sfreq=raw_7.info["sfreq"],
    orig_time=raw_7.info["meas_date"],
)
annot_from_events_8 = mne.annotations_from_events(
    events=events_8,
    event_desc=mapping_3,
    sfreq=raw_8.info["sfreq"],
    orig_time=raw_8.info["meas_date"],
)
raw_5.set_annotations(annot_from_events_5)
raw_6.set_annotations(annot_from_events_6)
raw_7.set_annotations(annot_from_events_7)
raw_8.set_annotations(annot_from_events_8)

# dictionary containing the channel names
channel_names = { 'ch_1': 'AF3', 'ch_2': 'FPz', 'ch_3': 'AF4', 'ch_4': 'F9', 'ch_5': 'F7', 'ch_6': 'FC4', 'ch_7': 'F10', 'ch_8': 'T7', 'ch_9': 'F5',
                  'ch_10': 'F3', 'ch_11': 'F1', 'ch_12': 'Fz', 'ch_13': 'F2', 'ch_14': 'F4', 'ch_15': 'F6', 'ch_16': 'F8', 'ch_17': 'CP5', 
                  'ch_18': 'FT7', 'ch_19': 'FC5', 'ch_20': 'FC3', 'ch_21': 'FC1', 'ch_22': 'FCz', 'ch_23': 'FC2', 'ch_24': 'FC6', 'ch_25': 'FT8', 
                  'ch_26': 'C2', 'ch_27': 'Cz', 'ch_28': 'C1', 'ch_29': 'POz', 'ch_30': 'CP2', 'ch_31': 'CP4', 'ch_32': 'CP6', 'ch_33': 'C6', 
                  'ch_34': 'T8', 'ch_35': 'TP7', 'ch_36': 'CP3', 'ch_37': 'CP1', 'ch_38': 'CPz', 'ch_39': 'Pz', 'ch_40': 'P4', 'ch_41': 'P2', 'ch_42': 'TP10', 
                  'ch_43': 'TP8', 'ch_44': 'P5', 'ch_45': 'P3', 'ch_46': 'P1', 'ch_47': 'PO3', 'ch_48': 'PO10', 'ch_49': 'P6', 'ch_50': 'P8', 'ch_51': 'PO4', 
                  'ch_52': 'P10', 'ch_53': 'P9', 'ch_54': 'P7', 'ch_55': 'PO7', 'ch_56': 'O2', 'ch_57': 'Oz', 'ch_58': 'PO9', 'ch_59': 'FT9', 'ch_60': 'PO8', 
                  'ch_61': 'C5', 'ch_62': 'FT10', 'ch_63': 'TP9', 'ch_64': 'O1'
}

# read channel locations from custom montage file
montage = mne.channels.read_custom_montage(fname='montage.xyz', coord_frame='head')
print(f"Imported montage information: {montage}")

# create a function to preprocess each raw data file
def preprocess(raw_data, channel_names, file_number):
    # print name of raw_data input file
    print(f"Preprocessing Raw {file_number}...")
    # change channel names to match the ones in the montage
    raw_data.rename_channels(channel_names)
    # apply the montage to the raw data
    raw_data.set_montage(montage)
    # apply a bandpass filter from 1Hz to 45Hz
    raw_data.filter(1, 45, method='iir')
    # plot the alpha section of the raw data
    raw_data.plot(
        n_channels=16,
        start=20,
        duration=10,
        color="darkblue",
        scalings={"eeg": 20},
        title=f"Raw {file_number} EEG data: Alpha section for outlier detection",
    )
    plt.show()
    # plot the raw data to select bad channels for interpolation
    raw_data.plot(
        n_channels=16,
        start=40,
        duration=30,
        color="darkblue",
        scalings={"eeg": 20},
        title=f"Raw {file_number} EEG data: Select bad channels for interpolation",
    )
    plt.show()
    print(raw_data.info) # shows which channels were marked as bad
    # interpolate bad channels
    raw_data.interpolate_bads(reset_bads=True)
    # plot the interpolated data
    raw_data.plot(
        n_channels=16,
        start=40,
        duration=30,
        color="darkblue",
        scalings={"eeg": 20},
        title=f"Raw {file_number} Interpolated data",
    )
    plt.show()
    # perform ICA to remove artifacts
    ica = mne.preprocessing.ICA(n_components=20, method='fastica', max_iter='auto')
    ica.fit(raw_data)
    ica
    # retrieve the fraction of variance in the original data that is explained by our ICA components
    explained_var_ratio = ica.get_explained_variance_ratio(raw_data)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
        )
    # plot the ICA components time series
    ica.plot_sources(raw_data, show_scrollbars=False, title="ICA components")
    ica.plot_components(sphere="eeglab", inst=raw_data)
    plt.show()
    # prompt the user to input the desired components to exclude
    exclude_input = input("Enter the indices of the components to exclude (separated by spaces): ")
    # convert the input string to a list of integers
    exclude_indices = [int(index) for index in exclude_input.split()]
    # validate the input: check if indices are within valid range
    if any(index < 0 or index >= ica.n_components for index in exclude_indices):
        raise ValueError("Invalid component indices provided. Please check the indices and try again.")
    # assign the exclude list to ica.exclude
    ica.exclude = exclude_indices
    # reconstruct the sensor signals with artifacts removed using the apply method
    rec_raw = raw_data.copy()
    ica.apply(rec_raw)
    # plot the original and reconstructed data
    raw_data.plot(
        n_channels=16,
        start=40,
        duration=10,
        color="darkblue",
        scalings={"eeg": 20},
        title=f"Raw {file_number} Original data",
    )
    rec_raw.plot(
        n_channels=16,
        start=40,
        duration=10,
        color="darkblue",
        scalings={"eeg": 20},
        title=f"Raw {file_number} ICA cleaned data",
    )
    plt.show()
    # return the cleaned data
    return rec_raw

# preprocess the raw data files
# rec_raw_1 = preprocess(raw_1, channel_names, file_number=1)
# rec_raw_1.save("subject_4/s4_t1_down_p1_cleaned.fif", overwrite=True)
# rec_raw_2 = preprocess(raw_2, channel_names, file_number=2)
# rec_raw_2.save("subject_4/s4_t1_down_p2_cleaned.fif", overwrite=True)
rec_raw_3 = preprocess(raw_3, channel_names, file_number=3)
rec_raw_3.save("subject_4/s4_t1_up_p1_cleaned.fif", overwrite=True)
rec_raw_4 = preprocess(raw_4, channel_names, file_number=4)
rec_raw_4.save("subject_4/s4_t1_up_p2_cleaned.fif", overwrite=True)
rec_raw_5 = preprocess(raw_5, channel_names, file_number=5)
rec_raw_5.save("subject_4/s4_t2_p1_cleaned.fif", overwrite=True)
rec_raw_6 = preprocess(raw_6, channel_names, file_number=6)
rec_raw_6.save("subject_4/s4_t2_p2_cleaned.fif", overwrite=True)
rec_raw_7 = preprocess(raw_7, channel_names, file_number=7)
rec_raw_7.save("subject_4/s4_t2_p3_cleaned.fif", overwrite=True)
rec_raw_8 = preprocess(raw_8, channel_names, file_number=8)
rec_raw_8.save("subject_4/s4_t2_p4_cleaned.fif", overwrite=True)

# load the cleaned data
rec_raw_1 = mne.io.read_raw_fif("subject_4/s4_t1_down_p1_cleaned.fif", preload=True)
rec_raw_2 = mne.io.read_raw_fif("subject_4/s4_t1_down_p2_cleaned.fif", preload=True)
rec_raw_3 = mne.io.read_raw_fif("subject_4/s4_t1_up_p1_cleaned.fif", preload=True)
rec_raw_4 = mne.io.read_raw_fif("subject_4/s4_t1_up_p2_cleaned.fif", preload=True)
rec_raw_5 = mne.io.read_raw_fif("subject_4/s4_t2_p1_cleaned.fif", preload=True)
rec_raw_6 = mne.io.read_raw_fif("subject_4/s4_t2_p2_cleaned.fif", preload=True)
rec_raw_7 = mne.io.read_raw_fif("subject_4/s4_t2_p3_cleaned.fif", preload=True)
rec_raw_8 = mne.io.read_raw_fif("subject_4/s4_t2_p4_cleaned.fif", preload=True)

# concatenate the raw objects together
raw_objects = [rec_raw_1, rec_raw_2, rec_raw_3, rec_raw_4, rec_raw_5, rec_raw_6, rec_raw_7, rec_raw_8]

# concatenate the raw objects
raw_all = mne.concatenate_raws(raw_objects)

# save the cleaned data
raw_all.save("subject_4/s4_preprocessed.fif", overwrite=True)




