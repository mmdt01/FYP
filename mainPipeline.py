from bin.MLEngine import MLEngine

if __name__ == "__main__":

    # load preprocessed data from subject 4
    dataset_details={
        'data_path' : "subject_4" ,
        'file_to_load': 's4_preprocessed.fif',
        'ntimes': 10,
        'kfold':10,
        'm_filters':2,
        'window_details':{'tmin':0,'tmax':3}
    }

    ML_experiment = MLEngine(**dataset_details)
    ML_experiment.experiment()
