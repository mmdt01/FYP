from bin.MLEngine import MLEngine

if __name__ == "__main__":
    # '''Example for loading Korea University Dataset'''
    # dataset_details = {
    #     'data_path': "/Volumes/Transcend/BCI/KU_Dataset/BCI dataset/DB_mat",
    #     'subject_id': 1,
    #     'sessions': [1],
    #     'ntimes': 1,
    #     'kfold': 10,
    #     'm_filters': 2,
    # }

    # '''Example for loading BCI Competition IV Dataset 2a'''
    # dataset_details={
    #     'data_path' : "data/BCICIV_2a_gdf" ,
    #     'file_to_load': 'A01T.gdf',
    #     'ntimes': 10,
    #     'kfold':10,
    #     'm_filters':2,
    #     'window_details':{'tmin':0.5,'tmax':2.5}
    # }

    '''Example for loading my data'''
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
