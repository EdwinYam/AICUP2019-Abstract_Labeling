import os
import pandas as pd


def save_split_set(model_config):

    dataset = pd.read_csv(os.path.join(model_config["data_path"],), 
                          dtype=str)
    dataset.drop()
    
