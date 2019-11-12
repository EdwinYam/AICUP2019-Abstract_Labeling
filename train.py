import os

from torch.utils.data import DataLoader
from transformers import AdamW, WarmupLinearSchedule


def check_data():
    BATCH_SIZE = 64
    trainset = 
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
    data = next(iter(trainloader))
    input_indices, segment_indices, input_mask, label = data
    print(input_indices, segment_indices, input_mask, label)
