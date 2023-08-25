import torch
from torch.optim import Adam
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


def get_optimizer_and_scheduler(model, lr, beta1, beta2, eps, warmup_steps, t_total):
    # For training Transformer
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    return optimizer, scheduler