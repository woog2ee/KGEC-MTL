import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from operator import itemgetter


class TransformerDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.src_lst = list(map(itemgetter(0), data))
        self.tgt_lst = list(map(itemgetter(1), data))
        self.tokenizer = tokenizer
        self.dataset = self.build_dataset(data)   

    def build_dataset(self, data):
        dataset = [(self.tokenizer.encode_src(src), self.tokenizer.encode_tgt(tgt))
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
    src_sent = pad_sequence([torch.tensor(src) for src, _ in batch_samples],
                             batch_first=True,
                             padding_value=pad_num)
    tgt_sent = pad_sequence([torch.tensor(tgt) for _, tgt in batch_samples],
                             batch_first=True,
                             padding_value=pad_num)
    return src_sent.to(device), tgt_sent.to(device)


def batch_sampling(seq_lengths, batch_size):
    # Compose batch to match similar lengths of sequences
    seq_lengths = [(i, seq_len, tgt_len) for i, (seq_len, tgt_len) in enumerate(seq_lengths)]
    seq_lengths = sorted(seq_lengths, key=lambda x: x[1])
    seq_lengths = [sample[0] for sample in seq_lengths]

    sample_indices = [seq_lens[i:i+batch_size] for i in range(0, len(seq_lengths), batch_size)]
    random.shuffle(sample_indices)
    return sample_indices


def build_data_loader(dataset, batch_size, num_workers, curriculum=False):
    if curriculum:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 sampler=SequentialSampler(dataset),
                                 collate_fn=collate_fn,
                                 num_workers=num_workers)
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 sampler=RandomSampler(dataset),
                                 collate_fn=collate_fn,
                                 num_workers=num_workers)
    return data_loader