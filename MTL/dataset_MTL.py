import torch
from torch.utils.data import Dataset


class MTLDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.dataset = list(zip([d[1] for d in data], [d[0] for d in data]))
        self.tokenizer = tokenizer
        self.pad_num = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        self.label_pad_num = -1

        self.with_pretrained = args.with_pretrained
        self.max_tokens_per_sent = args.max_tokens_per_sent
        
        
    def tokenize(self, sent, padding=True):
        if padding:
            # Return with input_ids, token_type_ids, and attention_mask
            tokenized = self.tokenizer(sent,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_tokens_per_sent)
        else:
            # Return with only tokens
            tokenized = self.tokenizer.tokenize(sent)
        return tokenized
        
        
    def get_word_level_labels(self, sent1, sent2):
        words1, words2 = sent1.split(' '), sent2.split(' ')
        labels = [1 if w1 != w2 else 0 for w1, w2 in zip(words1, words2)]
        return labels
                                                
    
    def get_token_level_labels(self, sent1, sent2, padding=True):
        word_level_labels = self.get_word_level_labels(sent1, sent2)
        each_word_token_lengths = self.get_each_word_token_lengths(sent1)
                                                
        labels = [[word_level_labels[i]] * each_word_token_lengths[i]
                  for i in range(len(word_level_labels))]
        labels = sum(labels, [])
                                                
        if padding:
            labels += [self.label_pad_num] * (self.max_tokens_per_sent - len(labels))
            assert len(labels) == self.max_tokens_per_sent
        return labels


    def get_each_word_token_lengths(self, sent):
        # '날씨가 좋아요' -> [['날씨', '##가'], ['좋아', '##요']] -> [2, 2]
        each_word_token_lengths = [len(self.tokenize(word, padding=False)) for word in sent.split(' ')]
        return each_word_token_lengths
                                        
        
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, idx):
        noised, origin = self.dataset[idx][0], self.dataset[idx][1]
                                        
        if self.with_pretrained:
            src = self.tokenize(noised)['input_ids']
            tgt = self.tokenize(origin)['input_ids']
        else:
            src = self.tokenizer.encode_src(src)
            tgt = self.tokenizer.encode_tgt(tgt) # 패딩 필요

        label = self.get_token_level_labels(noised, origin, padding=True)
        
        output = {'src': src, 'label': label, 'tgt': tgt}
        return {k: torch.tensor(v) for k, v in output.items()}