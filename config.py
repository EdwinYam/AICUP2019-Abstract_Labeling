import os
import numpy as np
from sacred import Ingredient

config_ingredient = Ingredient("cfg")

@config_ingredient.config
def cfg():
    model_config = {"data_path": "/media/Datasets/PaperAbstract/",
                    "log_dir": "logs",
                    "task": "classification", # classification, labeling
                    }
    experiment_id = np.random.randint(0,1e6)
    if model_config["task"] == "classification"
