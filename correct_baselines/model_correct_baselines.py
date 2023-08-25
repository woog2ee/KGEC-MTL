import torch
import torch.nn as nn
from torch.nn import Transformer
import math


class TokenEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([args.hidden_size])).to(args.device)
        
    def forward(self, x):
        return self.embedding(x) * self.scale


class PositionalEmbedding(nn.Module):
    def __init__(self, args, max_len=512):
        super().__init__()
        
        self.dropout = nn.Dropout(args.dropout)
        
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.hidden_size, 2) * (-math.log(10000) / args.hidden_size))
        
        pos_embedding = torch.zeros(max_len, 1, args.hidden_size)
        pos_embedding.require_grad = False
        
        pos_embedding[:, 0, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 0, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pos_embedding', pos_embedding)
        
    def forward(self, token_embedding):
        token_embedding += self.pos_embedding[:token_embedding.size(0), :]
        return self.dropout(token_embedding)
    

class TransformerCorrector(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.src_token_embedding = TokenEmbedding(args)
        self.tgt_token_embedding = TokenEmbedding(args)
        self.pos_embedding = PositionalEmbedding(args)
        
        self.transformer = Transformer(d_model=args.hidden_size,
                                       nhead=args.n_heads,
                                       num_encoder_layers=args.n_enc_layers,
                                       num_decoder_layers=args.n_dec_layers,
                                       dim_feedforward=args.pf_dim,
                                       dropout=args.dropout)
        self.linear = nn.Linear(args.hidden_size, args.vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask,
                src_pad_mask, tgt_pad_mask, memory_key_pad_mask):
        src_emb = self.src_token_embedding(src)
        src_emb = self.pos_embedding(src_emb)
        
        tgt_emb = self.tgt_token_embedding(tgt)
        tgt_emb = self.pos_embedding(tgt_emb)
        
        out = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask,
                               None,  # None for memory_mask
                               src_pad_mask, tgt_pad_mask, memory_key_pad_mask)
        return self.linear(out)
        
    def encode(self, src, src_mask):
        src_emb = self.src_token_embedding(src)
        src_emb = self.pos_embedding(src_emb)
        return self.transformer.encoder(src_emb, src_mask)
    
    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.tgt_token_embedding(tgt)
        tgt_emb = self.pos_embedding(tgt_emb)
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)
    
    
def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    src_mask    = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    tgt_seq_len = tgt.shape[0]
    tgt_mask    = generate_square_subsequent_mask(tgt_seq_len, device)

    pad_idx = 3  # Preprocessor's pad_token_id
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
            
            
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)