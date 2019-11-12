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
        model_config['task_name'] = 'Task 1'
        model_config['input_method'] = 'single'
        model_config['train_filename'] = 'task1_trainset.csv'
        model_config['test_filename'] = 'task1_public_testset.csv'
        model_config['labels'] = {'BACKGROUND':0, 'OBJECTIVES':1, 'METHODS':2, 'RESULTS':3, 'CONCLUSIONS':4, 'OTHERS':5}
    elif model_config['task'] == 'classification':
        model_config['task_name'] = 'Task 2'
        model_config['train_filename'] = 'task2_trainset.csv'
        model_config['test_filename'] = 'task2_public_testset.csv'
        model_config['labels'] = {'THEORETICAL':0, 'ENGINEERING':1, 'EMPIRICAL':2, 'OTHERS':3}


