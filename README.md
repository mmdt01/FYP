# Decoding Robot Interaction Tasks from EEG Signals

This repository does not contain the raw or preprocessed EEG files for the subjects. Please seperately download and move the preprocessed file for subject 4 (s4_preprocessed.fif) into the directory 'subject_4'.

## Preprocessing:

- preprocess.py outlines the full procedure of loading, cleaning and preparing the raw EEG data for classification

## Loading and Analysis

- load-eeg.py provides a demonstration of how a single subject's preprocessed data can be loaded and visualized in python.

## Classification

- This repository intends to train and test several classification models on each sybjects preprocessed EEG data. Currently, the only algorithms implemented are Filter Bank Common Spatial Patter (FBCSP) for feature extraction used to train a Support Vector Machine (SVM). This repository uses code found from the FBCSP Toolbox. To run the algorithm use mainPipeline.py.

