import os
import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    model_config = {"seed": 123456,
                    "data_path": "/media/Datasets/PaperAbstract/",
                    "log_dir": "logs",
                    "task": "classification", # classification, labeling
                    'num_workers': 4,
                    'drop_fields': ['id','Title','Categories','Created Date','Authors'],
                    }
    experiment_id = np.random.randint(0,1e6)
    if model_config['task'] == 'labeling':
        model_config['train_filename'] = 'task1_trainset.csv'
        model_config['test_filename'] = 'task1_public_testset.csv'
    elif model_config['task'] == 'classification':
        model_config['train_filename'] = 'task2_trainset.csv'
        model_config['test_filename'] = 'task2_public_testset.csv'

