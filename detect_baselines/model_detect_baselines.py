import torch
import torch.nn as nn


class BERTDetector(nn.Module):
    def __init__(self, args, bert):
        super().__init__()
        
        self.bert = bert
        for param in self.bert.parameters():
           param.requires_grad = False
        self.dropout = nn.Dropout(args.dropout)
        
        self.linear = nn.Linear(args.hidden_size, 2)


    def get_attn_mask(self, token_ids, valid_length):
        attn_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attn_mask[i][:v] = 1
        return attn_mask.float()
    
        
    def forward(self, input_ids, valid_length, segment_ids):
        # token_ids: [batch_size, max_length]

        attn_mask = self.get_attn_mask(input_ids, valid_length)
        
        out = self.bert(input_ids=input_ids,
                        token_type_ids=segment_ids.long(),
                        attention_mask=attn_mask.float().to(input_ids.device)).last_hidden_state
        out = self.dropout(out)
        # out: [batch_size, max_length, hidden_size]
        
        classified = self.linear(out)
        # classified: [batch_size, max_length, 2]
        
        return torch.sigmoid(classified)