import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
import json
import random
import numpy as np


def str2bool(str):
    if str == 'true':
        return True
    elif str == 'false':
        return False
    else:
        raise 'Invalid arguments, boolean value expected'
        
        
def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
        
def load_json_data(data_path):
    with open(data_path, 'r', encoding='UTF8') as f:
        data = json.load(f)
    return data
        
        
def get_model_and_tokenizer(model_path):
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_optimizer_and_scheduler(model, lr, warmup_steps, t_total):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    return optimizer, scheduler