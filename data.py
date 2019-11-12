import torch
from torch.utils.data import Dataset

class abstractSegment(object):
    def __init__(self, segment, pos_index, label=None):
        self.segment = segment
        self.pos_index = label
        self.label = label

class abstractFeature(object):
    ''' One input sample/instance '''
    def __init__(self,):

class AbstractDataset(Dataset):
    def __init__(self, model_config, tokenizer, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode
        self.df = pd.read_csv(mode+'.tsv', sep='\t').fillna('')
        self.label_map = model_config['labels']
        self.tokenizer = tokenizer
        self.data_list = None
        if model_config['task'] == 'labeling':
            data_list = self.__preprocess1__(model_config)
            self.data_list = convert_examples_to_features(model_config, data_list, tokenizer)
        elif model_config['task'] == 'classification':
            self.data_list = self.__preprocess2__(model_config)
        self.data_size = len(self.data_list)

    def __preprocess1__(self, model_config):
        # 1. Take single sentence as input
        # 2. Take the whole abstract as input if it doesn't exceed the
        #    predefined max length, else, split the whole abstract
        #    model_config['sen_range']=5
        # 3. Take the output from BERT as input embedding for simple-net

        data_list = list()
        
        if self.mode == 'train':
            for instance in df.iterrows():
                abstract=instance[0]['Abstract'].split('$$$')
                labels=instance[0][model_config['task_name']].split()
                for idx,sent in enumerate(abstract):
                    label_list.append(labels[idx])
                    pos_list.append(idx)
                    sen_width = model_config['sen_range'] // 2
                    if idx < sen_width:
                        data_list.append(
                            abstractSegment(
                                segment=abstract[:model_config['sen_range']],
                                pos_index=idx,
                                label=labels[idx]))
                    else:
                        data_list.append(
                            abstractSegment(
                                segment=abstract[idx-sen_width:idx+sen_width+1],
                                pos_index=idx,
                                label=labels[idx]))
        
        if self.mode == 'test':
            for instance in df.iterrows():
                abstract = instance[0]['Abstract'].split('$$$')
                for idx, sent in enumerate(abstract):
                    sen_width = model_config['sen_range'] // 2
                    if idx < sen_width:
                        data_list.append(
                            abstractSegment(
                                segment=abstract[:model_config['sen_range']],
                                pos_index=idx))
                    else:
                        data_list.append(
                            abstractSegment(
                                segment=abstract[idx-sen_width:idx+sen_width+1],
                                pos_index=idx))

        return data_list

    def __preprocess2__(self, model_config):
        pass
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        word_pieces = ['[CLS]']
        tokens = self.tokenizer.tokenize(data_list[idx])
        word_pieces += tokens + ['[SEP]']
        token_length = len(word_pieces)
        label_tensor = torch.tensor(label_to_onehot(self.label_list[idx])) if self.mode == 'train' else None


def label_to_onehot(labels):
    ''' Convert label to onehot vector
        Args:
            labels (str): labels of one sentence
        Return:
            outputs (onehot list): labels in the form of onehot list
    '''
    onehot = [0]*len(model_config['labels'])
    for l in labels.split('/'):
        onehot[model_config['labels']] = 1
    return onehot

def convert_examples_to_features(model_config, examples, tokenizer):
    unique_id = 1e8
    features = list()
    for (example_index, example) in enumerate(tqdm(examples)):
        text_tokens = tokenizer.tokenize(' '.join(example.segment))
        if len(text_tokens) > model_config['max_length']:
            print('[Error] Sentence length {} exceeds max length{}'.format(len(text_tokens),model_config['max_length']))
            raise ValueError
        

