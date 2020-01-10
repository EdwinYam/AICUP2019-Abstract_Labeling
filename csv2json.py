import pandas as pd
import os
import json
import random
random.seed(123)
# df = pd.read_csv('../task1_public_testset.csv')
# '../task1_sample_submission.csv'
df = pd.read_csv('../task1_trainset.csv')
df = df.drop(['Title','Authors','Categories','Created Date'], axis=1)

''' [BACKGROUND, OBJECTIVES, METHODS, RESULTS, CONCLUSIONS, OTHERS] '''

categories = ['BACKGROUND', 'OBJECTIVES', 'METHODS', 'RESULTS', 'CONCLUSIONS', 'OTHERS']

size = int(0.1*len(df))
dev_indices = random.sample(list(range(len(df))), size)

train_filename = 'data/PaperAbstract/train.jsonl'
dev_filename = 'data/PaperAbstract/dev.jsonl'
if not os.path.isfile(train_filename) and not os.path.isfile(dev_filename):
    train_file = open('data/PaperAbstract/train.jsonl', 'w')
    dev_file = open('data/PaperAbstract/dev.jsonl', 'w')
    for idx, instance in df.iterrows():
        data = dict()
        data['abstract_id'] = instance['Id']
        data['sentences'] = instance['Abstract'].split('$$$')
        labels = instance['Task 1'].split()
        data['labels'] = [ l.split('/') for l in labels ]
        
        #train_file.write(json.dumps(data))
        #train_file.write('\n')
        #dev_file.write(json.dumps(data))
        #dev_file.write('\n')
        
        # if idx not in dev_indices:
        #     train_file.write(json.dumps(data))
        #     train_file.write('\n')
        # else:
        #     dev_file.write(json.dumps(data))
        #     dev_file.write('\n')

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
            
            #train_file.write(json.dumps(data))
            #train_file.write('\n')
            #dev_file.write(json.dumps(data))
            #dev_file.write('\n')
            
            # if idx not in dev_indices:
            #     train_file.write(json.dumps(data))
            #     train_file.write('\n')
            # else:
            #     dev_file.write(json.dumps(data))
            #     dev_file.write('\n')

df = pd.read_csv('../task1_private_testset.csv')
df = df.drop(['Title','Authors','Categories','Created Date'], axis=1)
test_file = open('data/PaperAbstract/private_test.jsonl', 'w')
for idx, instance in df.iterrows():
    data = dict()
    data['abstract_id'] = instance['Id']
    data['sentences'] = instance['Abstract'].split('$$$')
    test_file.write(json.dumps(data))
    test_file.write('\n')
