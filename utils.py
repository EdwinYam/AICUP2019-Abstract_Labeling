import os
import pandas as pd
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def collect_words(model_config):
    

def save_split_set(model_config):

    dataset = pd.read_csv(os.path.join(model_config['data_path'], model_config['train_filename']), dtype=str)
    testset = pd.read_csv(os.path.join(model_config['data_path'], model_config['test_filename']), dtype=str)
    if model_config['drop_info']:
        dataset.drop('Title', axis=1, inplace=True)
        dataset.drop('Authors', axis=1, inplace=True)
        dataset.drop('Categories', axis=1, inplace=True)
        dataset.drop('Created Date', axis=1, inplace=True)

        testset.drop('Title', axis=1, inplace=True)
        testset.drop('Authors', axis=1, inplace=True)
        testset.drop('Categories', axis=1, inplace=True)
        testset.drop('Created Date', axis=1, inplace=True)
    
    trainset, validset = train_test_split(dataset, test_size=0.1, random_state=model_config['seed'])
    
