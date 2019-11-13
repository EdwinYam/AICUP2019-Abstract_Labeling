import torch
from torch.utils.data import Dataset

class abstractExample(object):
    ''' A single training/test example for abstract labeling/classification'''
    def __init__(self, segment, pos_index, label=None)
        ''' Constructs a input example for abstract
        Args:
            segment: list. The list of untokenized sentences, which is composed of the target sentence and few sentences before or after the target sentences.
            pos_index: int. The index of the target sentence within the range of sentences selected
            label: (Optional) list. The label of the example, in the form of an one-hot vector
        '''
        self.segment = segment
        self.pos_index = pos_index
        self.label = label

class abstractFeature(object):
    ''' One set of input features for one sample/instance '''
    def __init__(self, unique_id, example_index, tokens, input_indices, segment_indices, input_mask, start_position, end_position, label):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.input_indices = input_indices
        self.segment_indices = segment_indices
        self.input_mask = input_mask
        self.start_position = start_position
        self.end_position = end_position
        self.label = label

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
        onehot[label_map] = 1
    return onehot

def preprocess_for_labeling(model_config, df, mode='train'):
    # 1. Take single sentence as input
    # 2. Take the whole abstract as input if it doesn't exceed the
    #    predefined max length, else, split the whole abstract
    #    model_config['sen_range']=5
    # 3. Take the output from BERT as input embedding for simple-net

    data_list = list()
    
    if mode == 'train':
        for instance in df.iterrows():
            abstract=instance[0]['Abstract'].split('$$$')
            labels=instance[0][model_config['task_name']].split()
            for idx,sent in enumerate(abstract):
                label_list.append(labels[idx])
                pos_list.append(idx)
                sen_width = model_config['sen_range'] // 2
                if idx < sen_width:
                    data_list.append(
                        abstractExample(
                            segment=abstract[:model_config['sen_range']],
                            pos_index=idx,
                            label=labels[idx]))
                else:
                    data_list.append(
                        abstractExample(
                            segment=abstract[idx-sen_width:idx+sen_width+1],
                            pos_index=idx,
                            label=labels[idx]))
    
    if mode == 'test':
        for instance in df.iterrows():
            abstract = instance[0]['Abstract'].split('$$$')
            for idx, sent in enumerate(abstract):
                sen_width = model_config['sen_range'] // 2
                if idx < sen_width:
                    data_list.append(
                        abstractExample(
                            segment=abstract[:model_config['sen_range']],
                            pos_index=idx))
                else:
                    data_list.append(
                        abstractExample(
                            segment=abstract[idx-sen_width:idx+sen_width+1],
                            pos_index=idx))

    return data_list

def read_abstract_examples(model_config, input_file, is_training=True):
    '''
    Read a PaperAbstract csv file into a list of abstractExample
    ------------------------------------------------------------
    examples = read_abstract_examples(model_config,
                                      input_file=input_file,
                                      is_training=not evaluate)
    '''

    df = pd.read_csv(input_file, sep='\t').fillna('')
    mode = 'train' if is_training else 'test'
    data_list = preprocess_for_labeling(model_config, df, mode)
    examples = list()
    for data in data_list:
        # (segment, pos_index, label)
        if data.label != None:
            data.label = label_to_onehot(data.label, model_config['label_map'])
        
        example = abstractExample(
            segment=data.segment,
            pos_index=data.pos_index,
            label=data.label)
        examples.append(data)
    return examples


def convert_examples_to_features(model_config, examples, tokenizer):
    unique_id = 1e8
    features = list()
    for (example_index, example) in enumerate(tqdm(examples)):
        # Add special tokens to origin text
        word_pieces = ['[CLS]']
        for index, sentence in enumerate(example.segment):
            tokens = tokenizer.tokenize(sentence)
            start_position, end_position = None, None
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
        segment_indices = [0]*len(word_pieces)
        segment_indices[start_position:end_position+1] = [1]*(end_position-start_position+1)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if model_config['mask_padding_with_zero'] else 0] * len(word_indices)

        # Zero-pad up to the max sequence length
        while len(word_indices) < model_config['max_seq_length']:
            word_indices.append(model_config['pad_token'])
            segment_indices.append(model_config['pad_token_segment_id'])
        
        assert len(word_indices) == model_config['max_seq_length']
        assert len(segment_indices) == model_config['max_seq_length']
        features.append(
            AbstractFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=word_pieces,
                input_indices=word_indices,
                segment_indices=segment_indices,
                input_mask=input_mask,
                start_position=start_position,
                end_position=end_position,
                label=example.label
            ))
        unique_id += 1
    return features

        

def load_and_cacahe_examples(model_config, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_file = 'train.tsv' if evaluate else 'test.tsv'
    cached_features_file = os.path.join(model_config['data_path'], 
                                        'cached_{}_{}_{}'.format(
                                            'dev' if evaluate else 'train',
                                            str(model_config['model']),
                                            str(model_config['max_seq_length'])
                                        ))
    if os.path.exists(cached_features_file) and not model_config['overwrite_cache'] and not :
        print('[Loading data] Loading features from cached file {}'.format(cached_features_file))
        features = torch.load(cached_features_file)
    else: 
        print('[Loading data] Creating features from dataset files at {}'.format(input_file))
        examples = read_abstract_examples(input_file=input_file,
                                          is_training=not evaluate,
                                          version_2_with_negative=model_config['version_2_with_negative'])
        features = convert_examples_to_features(model_config,
                                                examples=examples,
                                                tokenizer=tokenizer)
        print('[Dataset] Saving features into cached file {}'.format(cached_features_file))
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_indices = torch.tensor([f.input_indices for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_indices = torch.tensor([f.segment_indices for f in features], dtype=torch.long)
    if evaluate:
        dataset = TensorDataset(all_input_indices, all_input_mask, all_segment_indices)
    else:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_indices, all_input_mask, all_segment_indices, all_labels)

    if output_examples:
        return dataset, examples, features
    return dataset
