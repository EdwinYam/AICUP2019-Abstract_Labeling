import os
import pandas as pd
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def collect_words(df, model_config):
    sentences = list()
    for i in df.iterrows():
        sentences += i[0]['Abstract'].split('$$$')
    chunks = [
        ' '.join(sentences[i:i+len(sentence)//model_config['num_workers']])
    ]



def save_split_set(model_config):

    dataset = pd.read_csv(os.path.join(model_config['data_path'], model_config['train_filename']), dtype=str)
    testset = pd.read_csv(os.path.join(model_config['data_path'], model_config['test_filename']), dtype=str)
    
    if model_config['drop_info']:
        for field in model_config["drop_fields"]:
            dataset.drop(field, axis=1, inplace=True)
            testset.drop(field, axis=1, inplace=True)
    
    ''' Split data to training and validation set '''
    trainset, validset = train_test_split(dataset, test_size=0.1, random_state=model_config['seed'])
    print('[data_info] Train set size: {}'.format())
    print('[data_info] Test set size: {}'.format())

    ''' Extract all word types '''


        
