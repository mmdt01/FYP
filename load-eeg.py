# This python script loads the preprocessed EEG data and:
#   - plots the raw data
#   - extracts events from the annotations
#   - extracts epochs
#   - plots the epochs
#   - performs some analysis on the epochs

# import the necessary modules
import mne
import matplotlib.pyplot as plt
import random
import numpy as np

# define data path
data_path = "subject_4"
file_name = "s4_preprocessed.fif"

# create an event dictionary
event_dict = {
    'motor execution up': 1,
    'motor execution down': 2,
    'visual perception up': 3,
    'visual perception down': 4,
    'imagery up': 5,
    'imagery down': 6,
    'imagery and perception up': 7,
    'imagery and perception down': 8
}

# load the preprocessed data
raw = mne.io.read_raw_fif(data_path + "/" + file_name, preload=True)
events, event_ids = mne.events_from_annotations(raw, event_id=event_dict)

# print the events and event IDs
print("Events =", events)
print("Event IDs =", event_ids)

# plot the raw data
raw.plot(
    events=events,
    n_channels=16,
    start=40,
    duration=50,
    color="darkblue",
    scalings={"eeg": 20},
    title="Final Concatenated Preprocessed Data",
)
plt.show()

# print the sampling frequency
fs = raw.info.get('sfreq')
print(fs)

# take epochs of only events with the event IDs 3 and 4
epochs = mne.Epochs(
    raw,
    events,
    event_id=[3, 4],
    tmin=0,
    tmax=3,
    baseline=None,
    preload=True,
)
print(epochs)
print(len(epochs))

# shuffle the epochs
permutation = np.random.permutation(len(epochs))
print("permutation:", permutation)
epochs = epochs[permutation]
print(epochs)
print(len(epochs))

# plot the epochs
epochs.plot(n_channels=16, scalings={"eeg": 20}, title="Epochs", n_epochs=5, events=True)
plt.show()
print(epochs.drop_log) # shows which epochs were dropped


# # epoch the data based on the events: 3 second window after the event trigger
# epochs = mne.Epochs(
#     raw,
#     events,
#     event_id=event_dict,
#     tmin=0,
#     tmax=3,
#     baseline=None,
#     preload=True,
# )
# print(epochs)
# print(len(epochs))

# # plot the epochs
# epochs.plot(n_channels=16, scalings={"eeg": 20}, title="Epochs", n_epochs=5, events=True)
# plt.show()
# # print the epochs that were dropped
# print(epochs.drop_log) 

# extract the labels from the events using the last column of the events array
labels = epochs.events[:, -1]
print(labels)






# # Some analysis on the epochs for example, plot the power spectrum of imagery epochs

# # plot the power spectrum of imagery epochs
# imagery_up_epochs = epochs["imagery up"]
# perception_up_epochs = epochs["visual perception up"]
# imagery_up_spectrum = imagery_up_epochs.compute_psd(fmin=1, fmax=45).plot(sphere="eeglab")
# perception_up_spectrum = perception_up_epochs.compute_psd(fmin=1, fmax=45).plot(sphere="eeglab")
# plt.show()

# # plot epochs as an image map
# imagery_up_epochs.plot_image(picks=["PO3", "POz", "PO4"], combine='mean', title="Imagery Up Mean Epochs Image: PO3, POz, PO4")
# perception_up_epochs.plot_image(picks=["PO3", "POz", "PO4"], combine='mean', title="Perception Up Mean Epochs Image: PO3, POz, PO4")
# plt.show()

# # plot the topomap of the average of imagery epochs
# imagery_up_evoked = imagery_up_epochs.average() 
# imagery_up_evoked.plot_topomap(sphere="eeglab")
# perception_up_evoked = perception_up_epochs.average()
# perception_up_evoked.plot_topomap(sphere="eeglab")
# plt.show()




