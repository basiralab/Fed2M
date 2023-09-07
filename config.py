CONFIG = {
        "CBT_resolution": 35,
        "fed_rounds": 20,
        "epochs_per_round": 10,
        "n_folds": 4,
        'batch_size': 5,
        "n_clusters": 2,
        "local_sample_size": 10,
        "global_sample_size": 20,
        "sampling_method": 'hierarchical',
        "early_stop": True,
        "freeze": True, 
        "ablated": True,
        "client1":{ "N_ROIs": 35,
                "learning rate": 0.00005,
                "lambda1": 2,  #topology
                "lambda2":0.5, # local centeredness
                "lambda3":0.2}, 
        "client2":{"N_ROIs": 160,
                "learning rate": 0.00005,
                "lambda1": 2,  
                "lambda2":0.5, # local centeredness
                "lambda3":0.2}, 
        "client3":{"N_ROIs": 268,
                "learning rate": 0.00005,
                "lambda1": 2,  
                "lambda2":0.5, # local centeredness
                "lambda3":0.2}, 
        "model_name": "test"       # to be edited

}

MODEL_PARAMS = {
        "dropout_prob": 0.5,
        "CBT_resolution": 35,
        "Modality1":{   "N_ROIs": 35,
                      "n_features": 595},
        "Modality2":{   "N_ROIs": 160,
                        "n_features": 12720},
        "Modality3":{   "N_ROIs": 268,
                        "n_features": 35778},
        "intermediate_resolution": 1000,
        "intermediate_channel": 1190
}