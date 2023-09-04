import torch
import numpy as np
from sklearn import metrics


def tensor2lst(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        return tensor
    
    
def get_lst_dim(lst):
    dim = []
    while isinstance(lst, list):
        dim.append(len(lst))
        lst = lst[0] if len(lst) > 0 else None
    return len(dim)


def compute_metrics(y_true, y_pred): 
    # Convert tensor to list type
    y_true = tensor2lst(y_true)
    y_pred = tensor2lst(y_pred)
    
    # Get dimension of lists
    dim_true = get_lst_dim(y_true)
    dim_pred = get_lst_dim(y_pred)
    
    # If each list has dimension over 2, concatenate
    if dim_true == dim_pred == 1:
        pass
    
    elif dim_true == dim_pred > 1:
        y_true = sum(y_true, [])
        y_pred = sum(y_pred, [])
        
    else:
        assert 'list y_true and y_pred have different dimensions'

    # Ignore pad_num -1 in label
    nopad_indices = np.where(np.array(y_true) != -1)[0].tolist()
    y_true = np.array(y_true)[nopad_indices].tolist()
    y_pred = np.array(y_pred)[nopad_indices].tolist()
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    
    return accuracy, precision, recall, f1