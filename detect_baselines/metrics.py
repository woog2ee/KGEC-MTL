import torch
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


def combine_lst(lst):
    while(get_lst_dim(lst)) > 1:
        lst = sum(lst, [])
    return lst


def compute_metrics(y_true, y_pred):
    y_true = tensor2lst(y_true)
    y_pred = tensor2lst(y_pred)
    
    dim_true = get_lst_dim(y_true)
    dim_pred = get_lst_dim(y_pred)
    
    if dim_true == dim_pred:
        y_true = combine_lst(y_true)
        y_pred = combine_lst(y_pred)
        
    else:
        assert 'list y_true and y_pred have different dimensions'
        
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    return accuracy, precision, recall, f1
