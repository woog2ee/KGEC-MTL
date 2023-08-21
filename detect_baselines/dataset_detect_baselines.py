import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        self.dataset = list(zip([d[1] for d in data], [d[0] for d in data]))
        self.tokenizer = tokenizer
        self.pad_num = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        
        self.max_tokens_per_sent = args.max_tokens_per_sent
        self.max_words_per_sent = args.max_words_per_sent
        self.detect_level = args.detect_level
        
        
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
        
        
    def get_word_level_labels(self, sent1, sent2, padding=False):
        words1, words2 = sent1.split(' '), sent2.split(' ')
        labels = [1 if w1 != w2 else 0 for w1, w2 in zip(words1, words2)]
        
        if padding:
            labels += [self.pad_num] * (self.max_words_per_sent - len(labels))
            assert len(labels) == self.max_words_per_sent
        return labels
                                                
    
    def get_token_level_labels(self, sent1, sent2, padding=True):
        word_level_labels = self.get_word_level_labels(sent1, sent2, padding=False)
        each_word_token_lengths = self.get_each_word_token_lengths(sent1, padding=False)
                                                
        labels = [[word_level_labels[i]] * each_word_token_lengths[i]
                  for i in range(len(word_level_labels))]
        labels = sum(labels, [])
                                                
        if padding:
            labels += [self.pad_num] * (self.max_tokens_per_sent - len(labels))
            assert len(labels) == self.max_tokens_per_sent
        return labels
                                        
    
    def get_each_word_token_lengths(self, sent, padding=True):
        # '날씨가 좋아요' -> [['날씨', '##가'], ['좋아', '##요']] -> [2, 2]
        each_word_token_lengths = [len(self.tokenize(word, padding=False)) for word in sent.split(' ')]
                                        
        if padding:
            each_word_token_lengths += [self.pad_num] * (self.max_words_per_sent - len(each_word_token_lengths))
            assert len(each_word_token_lengths) == self.max_words_per_sent
        return each_word_token_lengths
        
        
        
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, idx):
        noised, origin = self.dataset[idx][0], self.dataset[idx][1]
                                        
        tokenized = self.tokenize(noised)
        input_ids = tokenized['input_ids']
                                        
        if self.detect_level == 'token':
            label = self.get_token_level_labels(noised, origin, padding=True)
        elif self.detect_level == 'word':
            label = self.get_word_level_labels(noised, origin, padding=True)
                                        
        each_word_token_lengths = self.get_each_word_token_lengths(noised, padding=True)
                                        
        output = {'input_ids': input_ids, 'label': label, 'each_word_token_lengths': each_word_token_lengths}
        return {k: torch.tensor(v) for k, v in output.items()}