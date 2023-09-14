import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from operator import itemgetter
from tqdm import tqdm
from transformers import AutoTokenizer


class TransformerDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.dataset = list(zip([d[1] for d in data], [d[0] for d in data]))
        #self.src_lst = list(map(itemgetter(1), data))
        #self.tgt_lst = list(map(itemgetter(0), data))
        self.tokenizer = tokenizer
        self.sos_num = tokenizer.sos_num
        self.eos_num = tokenizer.eos_num
        self.pad_num = tokenizer.pad_num
        self.max_tokens_per_sent = args.max_tokens_per_sent
        #self.dataset = self.build_dataset()   

    # def build_dataset(self):
    #     dataset = [[torch.tensor(self.tokenizer.encode_src(src)),
    #                 torch.tensor(self.tokenizer.encode_tgt(tgt))]
    #                 for src, tgt in tqdm(zip(self.src_lst, self.tgt_lst), total=len(self.src_lst))]            
    #     return dataset

    def pad(self, encoded):
        if len(encoded) < self.max_tokens_per_sent:
            encoded += [self.pad_num] * (self.max_tokens_per_sent - len(encoded))
        return encoded
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        noised, origin = self.dataset[idx][0], self.dataset[idx][1]

        src = self.tokenizer.encode_src(noised)
        tgt = [self.sos_num] + self.tokenizer.encode_tgt(origin) + [self.eos_num]
        src, tgt = self.pad(src), self.pad(tgt)

        output = {'src': src, 'tgt': tgt}
        return {k: torch.tensor(v) for k, v in output.items()}


def build_data_loader(dataset, args, dtype='train'):
    if args.curriculum and dtype == 'train':
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(dataset),
                                 num_workers=args.num_workers)
    elif not args.curriculum and dtype == 'train':
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size,
                                 sampler=RandomSampler(dataset),
                                 num_workers=args.num_workers)
    elif dtype != 'train':
        data_loader = DataLoader(dataset,
                                 batch_size=args.batch_size if dtype != 'test' else 1,
                                 sampler=RandomSampler(dataset),
                                 num_workers=args.num_workers)
    return data_loader