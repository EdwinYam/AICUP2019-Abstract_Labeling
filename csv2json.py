import pandas as pd
import os
import sys
import json
import random
random.seed(123)
categories = ['BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'OTHERS']

# task1_trainset.csv / task1_public_testset.csv / task1_private_testset.csv

def csv2json(data_path, split_ratio=0.0):
    df = pd.read_csv(data_path)
    df = df.drop(['Title','Authors','Categories','Created Date'], axis=1)

    size = int(split_ratio*len(df))
    dev_indices = random.sample(list(range(len(df))), size)

    train_filename = 'data/PaperAbstract/train.jsonl'
    dev_filename = 'data/PaperAbstract/dev.jsonl'
    if not os.path.exists('data/PaperAbstract'):
        os.makedirs('data/PaperAbstract')

    if not os.path.isfile(train_filename) and not os.path.isfile(dev_filename): 
        train_file = open('data/PaperAbstract/train.jsonl', 'w')
        dev_file = open('data/PaperAbstract/dev.jsonl', 'w')
        for idx, instance in df.iterrows():
            data = dict()
            data['abstract_id'] = instance['Id']
            data['sentences'] = instance['Abstract'].split('$$$')
            labels = instance['Task 1'].split()
            data['labels'] = [ l.split('/') for l in labels ]
        
            if int(split_ratio) > 0:
                if idx not in dev_indices:
                    train_file.write(json.dumps(data))
                    train_file.write('\n')
                else:
                    dev_file.write(json.dumps(data))
                    dev_file.write('\n')
            else:
                train_file.write(json.dumps(data))
                train_file.write('\n')
                dev_file.write(json.dumps(data))
                dev_file.write('\n')

    for category in categories:
        train_filename = 'data/PaperAbstract/train_{}.jsonl'.format(category)
        dev_filename = 'data/PaperAbstract/dev_{}.jsonl'.format(category)
        if not os.path.isfile(train_filename) and not os.path.isfile(dev_filename):
            train_file = open(train_filename, 'w')
            dev_file = open(dev_filename, 'w')
            for idx, instance in df.iterrows():
                data = dict()
                data['abstract_id'] = instance['Id']
                data['sentences'] = instance['Abstract'].split('$$$')
                labels = instance['Task 1'].split()
                data['labels'] = [ category if category in l else 'NONE' for l in labels ]
            
                if int(split_ratio) > 0:
                    if idx not in dev_indices:
                        train_file.write(json.dumps(data))
                        train_file.write('\n')
                    else:
                        dev_file.write(json.dumps(data))
                        dev_file.write('\n')
                else:
                    train_file.write(json.dumps(data))
                    train_file.write('\n')
                    dev_file.write(json.dumps(data))
                    dev_file.write('\n')

def csv2json_test(data_path):
    df = pd.read_csv(data_path)
    df = df.drop(['Title','Authors','Categories','Created Date'], axis=1)
    test_file = open('data/PaperAbstract/private_test.jsonl', 'w') if 'private' in data_path else open('data/PaperAbstract/public_test.jsonl', 'w')

    for idx, instance in df.iterrows():
        data = dict()
        data['abstract_id'] = instance['Id']
        data['sentences'] = instance['Abstract'].split('$$$')
        test_file.write(json.dumps(data))
        test_file.write('\n')

if  __name__ == '__main__':
    data_path = sys.argv[1]
    mode = sys.argv[2] # train or test
    if mode == 'train':
        csv2json(data_path)
    elif mode == 'test':
        csv2json_test(data_path)

