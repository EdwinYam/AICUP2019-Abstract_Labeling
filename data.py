import torch
from torch.utils.data import Dataset

from utils import InputExample

class AbstractExample(InputExample):
    ''' A single training/test example for abstract labeling/classification'''
    def __init__(self, guid, segment, pos_index, label=None, text_a=None, text_b=None):
        super(AbstractExample, self).__init__(guid, text_a, text_b, label)
        ''' Constructs a input example for abstract
        Args:
            segment: list. The list of untokenized sentences, which is composed 
            of the target sentence and few sentences before or after the target 
            sentences.
            pos_index: int. The index of the target sentence within the range of
            sentences selected
            label: (Optional) list. The label of the example, in the form of an 
            one-hot vector
        '''
        self.segment = segment
        self.pos_index = pos_index
        self.label = label

class AbstractFeature(InputFeature):
    ''' One set of input features for one sample/instance '''
    def __init__(self, instance_index, tokens, input_ids, token_type_ids, attention_mask, start_position=None, end_position=None, label=None):
        super(AbstractFeature, self).__init__(input_ids, attention_mask, token_type_ids, label)
        '''
        A single set of features of data.
        * A sequence of setences with the middle sentence as the target sentence 
        and the other sentences as context. 
        * input_ids[start_position:end_position] as the target sentence
        * tokens are words or subwords without converting into indices
        '''
        self.instance_index = instance_index
        self.tokens = tokens
        self.start_position = start_position
        self.end_position = end_position

class AbstractDataset(Dataset):
    def __init__(self, model_config, tokenizer, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode
        self.label_map = model_config['label_map']
        self.tokenizer = tokenizer
        self.data_list = None
        if model_config['task'] == 'labeling':
            examples = read_abstract_examples(model_config, 
                                              input_file=mode+'.tsv', 
                                              is_training=(mode=='train'))
            self.data_list = convert_examples_to_features(model_config, examples, tokenizer)
        elif model_config['task'] == 'classification':
            examples = read_abstract_examples(model_config, 
                                              input_file=mode+'.tsv', 
                                              is_training=(mode=='train'))
            self.data_list = convert_examples_to_features(model_config, examples, tokenizer)
        
        self.data_size = len(self.data_list)
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.data_list[idx]       

def label_to_onehot(labels, label_map):
    ''' Convert label to onehot vector
        Args:
            labels (str): labels of one sentence
        Return:
            outputs (onehot list): labels in the form of onehot list
    '''
    onehot = [0]*len(label_map)
    for l in labels.split('/'):
        onehot[label_map[l]] = 1.0
    return onehot

def preprocess_for_labeling(model_config, df, mode='train'):
    # 1. Take single sentence as input
    # 2. Take the whole abstract as input if it doesn't exceed the
    #    predefined max length, else, split the whole abstract
    #    model_config['sen_range']=5
    # 3. Take the output from BERT as input embedding for simple-net

    data_list = list()
    
    for i, instance in enumerate(df.iterrows()):
        abstract = instance[0]['Abstract'].split('$$$')
        labels = instance[0][model_config['task_name']].split() if mode == 'train' else [None]*len(abstract)

        for idx, sent in enumerate(abstract):
            sent_width = model_config['sent_range'] // 2
            if idx < sent_width:
                label_list = labels[:model_config['sent_range']]
                data_list.append(
                    AbstractExample(
                        guid='{}-{}'.format(i, idx),
                        segment=abstract[:model_config['sent_range']],
                        pos_index=idx,
                        label=labels[idx]))
            else:
                label_list = labels[idx-sent_width:idx+sent_width+1]
                data_list.append(
                    AbstractExample(
                        guid='{}-{}'.format(i, idx),
                        segment=abstract[idx-sent_width:idx+sent_width+1],
                        pos_index=idx,
                        label=labels[idx]))   
    return data_list

def preprocess_abstract_examples(model_config, df, mode='train'):
    data_list = list()
    
    for idx, instance in enumerate(df.iterrows()):
        abstract = instance[0]['Abstract'].split('$$$')
        labels = instance[0][model_config['task_name']].split() if mode == 'train' else [None]*len(abstract)
        labels = [ label_to_onehot(l) in for l in labels ]
        data_list.append(
            AbstractExample(
                guid=idx,
                segment=abstract,
                pos_index=None,
                label=labels))

    return data_list

def read_abstract_examples(model_config, input_file, is_training=True):
    '''
    Read a Paper Abstract csv file into a list of abstractExample
    ------------------------------------------------------------
    examples = read_abstract_examples(model_config,
                                      input_file=input_file,
                                      is_training=not evaluate)
    '''

    df = pd.read_csv(input_file, sep='\t').fillna('')
    mode = 'train' if is_training else 'test'
    examples = preprocess_for_labeling(model_config, df, mode)
    if is_training:
        for index in range(len(examples)):
            # (segment, pos_index, label)
            examples[index].label = label_to_onehot(examples[index].label, 
                                                    model_config['label_map'])
        return examples
    else:
        return examples


def convert_examples_to_features(model_config, examples, tokenizer):
    features = list()
    for (example_index, example) in enumerate(tqdm(examples, leave=False)):
        # Add special tokens to origin text
        word_pieces = ['[CLS]']
        start_position, end_position = 0, 0
        for index, sentence in enumerate(example.segment):
            tokens = tokenizer.tokenize(sentence)
            # TODO: adjust the way of adding speical tokens and assign segment
            #       index to each special token
            if index == example.pos_index:
                start_position = len(word_pieces)
                if index==0:
                    word_pieces += tokens + ['[SEP]']
                elif index==len(example.segment)-1:
                    word_pieces += ['SEP'] + tokens
                else:
                    word_pieces += ['SEP'] + tokens + ['SEP']
                end_position = len(word_pieces) - 1
            else:
                word_pieces += tokens
            
        word_indices = tokenizer.convert_tokens_to_ids(word_pieces)
        token_type_ids = [0]*len(word_pieces)
        token_type_ids[start_position:end_position+1] = [1]*(end_position-start_position+1)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if model_config['mask_padding_with_zero'] else 0] * len(word_indices)

        # Zero-pad up to the max sequence length
        while len(word_indices) < model_config['max_seq_length']:
            word_indices.append(model_config['pad_token'])
            token_type_ids.append(model_config['pad_token_segment_id'])
        
        assert len(word_indices) == model_config['max_seq_length']
        assert len(token_type_ids) == model_config['max_seq_length']
        features.append(
            AbstractFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=word_pieces,
                input_ids=word_indices,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                start_position=start_position,
                end_position=end_position,
                label=example.label
            ))
        unique_id += 1
    return features

        

def load_and_cache_examples(model_config, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_file = 'train.tsv' if evaluate else 'test.tsv'
    cached_features_file = os.path.join(model_config['data_path'], 
                                        'cached_{}_{}_{}'.format(
                                            'dev' if evaluate else 'train',
                                            str(model_config['model']),
                                            str(model_config['max_seq_length'])
                                        ))
    if os.path.exists(cached_features_file) and not model_config['overwrite_cache']:
        print('[Loading data] Loading features from cached file {}'.format(cached_features_file))
        features = torch.load(cached_features_file)
    else: 
        print('[Loading data] Creating features from dataset files at {}'.format(input_file))
        examples = read_abstract_examples(model_config=model_config,
                                          input_file=input_file,
                                          is_training=not evaluate)
        features = convert_examples_to_features(model_config,
                                                examples=examples,
                                                tokenizer=tokenizer)
        print('[Dataset] Saving features into cached file {}'.format(cached_features_file))
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if evaluate:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    else:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    if output_examples:
        return dataset, examples, features
    return dataset
