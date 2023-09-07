import re
from tqdm import tqdm


def sent2words(sent, aware_order=True):
    if aware_order:
        return [sent.split(' ')[i] + str(i) for i in range(len(sent.split(' ')))]
    else:
        return [sent.split(' ')[i] for i in range(len(sent.split(' ')))]

def sent2chars(sent, aware_order=True, aware_space=True):
    if aware_space:
        pass
    else:
        sent = re.sub(r' ', '', sent)
        
    if aware_order:
        return [sent[i] + str(i) for i in range(len(sent))]
    else:
        return [sent[i] for i in range(len(sent))]
    
def get_intersection(a, b):
    return list(set(a).intersection(b))

def compute_correction_metrics_persent(sent_pred, sent_gold,
                                       offset='word', aware_order=True, aware_space=True):
    assert type(sent_pred) == type(sent_gold) == str
    
    if offset == 'word':
        pred = sent2words(sent_pred, aware_order)
        gold = sent2words(sent_gold, aware_order)
    elif offset == 'char':
        pred = sent2chars(sent_pred, aware_order, aware_space)
        gold = sent2chars(sent_gold, aware_order, aware_space)
    else:
        assert 'wrong offset passed'
        
    precision = len(get_intersection(pred, gold)) / len(pred) if len(pred) != 0.0 else 0.0
    recall = len(get_intersection(pred, gold)) / len(gold) if len(gold) != 0.0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0.0 else 0.0
    return precision, recall, f1


def get_list_average(lst):
    return sum(lst) / len(lst)

def compute_correction_metrics_perlist(sent_pred_lst, sent_gold_lst,
                                       offset='word', aware_space=True):
    assert type(sent_pred_lst) == type(sent_gold_lst) == list
    
    precision_lst = []
    recall_lst = []
    f1_lst = []
    for i in tqdm(range(len(sent_pred_lst))):
        p, r, f1 = compute_correction_metrics_persent(sent_pred_lst[i], sent_gold_lst[i],
                                                      offset, aware_space)
        
        precision_lst.append(p)
        recall_lst.append(r)
        f1_lst.append(f1)
        
    return get_list_average(precision_lst), get_list_average(recall_lst), get_list_average(f1_lst)