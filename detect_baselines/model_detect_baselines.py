import torch
import torch.nn as nn


class BERTClassifier(nn.Module):
    def __init__(self, args, bert):
        super().__init__()
        
        self.bert = bert
        self.dropout = nn.Dropout(args.dropout)
        
        if args.linear_num == 1:
            self.linear_num = 1
            self.linear = nn.Linear(args.hidden_size, 2)
        elif args.linear_num == 2:
            self.linear_num = 2
            self.linear1 = nn.Linear(args.hidden_size, args.hidden_size//2)
            self.linear2 = nn.Linear(args.hidden_size//2, 2)
        
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):
        # token_ids: [batch_size, max_length]
        
        out = self.bert(input_ids).last_hidden_state
        out = self.dropout(out)
        # out: [batch_size, max_length, hidden_size]
        
        if self.linear_num == 1:
            classified = self.linear(out)
        elif self.linear_num == 2:
            out = self.linear1(out)
            classified = self.linear2(out)
        # classified: [batch_size, max_length, 2]
        
        return torch.sigmoid(classified)