import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from operator import itemgetter


class TransformerDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.src_lst = list(map(itemgetter(0), data))
        self.tgt_lst = list(map(itemgetter(1), data))
        self.tokenizer = tokenizer
        self.dataset = self.build_dataset()   

    def build_dataset(self):
        dataset = [[self.tokenizer.encode_src(src), self.tokenizer.encode_tgt(tgt)]
                    for src, tgt in zip(self.src_lst, self.tgt_lst)]
        return dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    
def collate_fn(batch_samples):
    # Sequence padding in batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_num = 2
    #print(f'batch_samples: {batch_samples}')
    src_sent = pad_sequence([torch.tensor(batch_samples[0])],
                             batch_first=True,
                             padding_value=pad_num)
    tgt_sent = pad_sequence([torch.tensor(batch_samples[1])],
                             batch_first=True,
                             padding_value=pad_num)
    return src_sent.to(device), tgt_sent.to(device)

def collate_fn_test(batch_samples):
    # Sequence padding in batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_num = 2
    #print(f'batch_samples: {batch_samples}')
    src_sent = pad_sequence([torch.tensor(batch_samples[0][0])],
                             batch_first=True,
                             padding_value=pad_num)
    tgt_sent = pad_sequence([torch.tensor(batch_samples[0][1])],
                             batch_first=True,
                             padding_value=pad_num)
    return src_sent.to(device), tgt_sent.to(device)


def build_data_loader(dataset, args, dtype='train'):
    if args.curriculum and dtype == 'train':
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(dataset),
                                 collate_fn=collate_fn if not test else collate_fn_test,
                                 num_workers=args.num_workers)
    elif not args.curriculum and dtype == 'train':
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 sampler=RandomSampler(dataset),
                                 collate_fn=collate_fn if not test else collate_fn_test,
                                 num_workers=args.num_workers)
    elif dtype != 'train':
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size if dtype != 'test' else 1,
                                 sampler=RandomSampler(dataset),
                                 collate_fn=args.collate_fn if not test else collate_fn_test,
                                 num_workers=num_workers)
    return data_loader