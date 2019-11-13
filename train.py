import os

from torch.utils.data import DataLoader
from transformers import AdamW, WarmupLinearSchedule
from transformers import BertForSequenceClassification


def set_seed(model_config):
    random.seed(model_config['seed'])
    np.random.seed(model_config['seed'])
    torch.manual_seed(model_config['seed'])

def check_data():
    BATCH_SIZE = 64
    tokenizer = 
    trainset = load_and_cache_examples(model_config, tokenizer, False)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
    data = next(iter(trainloader))
    input_indices, segment_indices, input_mask, label = data
    print(input_indices, segment_indices, input_mask, label)


def train():
