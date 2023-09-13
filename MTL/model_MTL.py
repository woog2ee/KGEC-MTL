import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import TransformerDecoderLayer, TransformerDecoder
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


class MultiTaskLearner(nn.Module):
    def __init__(self, args, bert):
        super().__init__()

        self.with_pretrained = args.with_pretrained

        # Token Embedding Layers for Shared Layers (with First-Initialized Transformer Encoder)
        self.src_token_embedding = TokenEmbedding(args)
        self.tgt_token_embedding = TokenEmbedding(args)
        self.pos_embedding = PositionalEmbedding(args)
        
        # Shared Layers (with Pretrained BERT)
        if self.with_pretrained:
            self.shared_layers = bert
            if args.freeze_layers == 0:
                pass
            else:
                bert_layers = ['embeddings.', 'encoder.layer.0.', 'encoder.layer.1.',
                               'encoder.layer.2.', 'encoder.layer.3.', 'encoder.layer.4.',
                               'encoder.layer.5.', 'encoder.layer.6.', 'encoder.layer.7.',
                               'encoder.layer.8.', 'encoder.layer.9.', 'encoder.layer.10.',
                               'encoder.layer.11.']

                for name, param in self.shared_layers.named_parameters():
                    if any(bert_layer in name for bert_layer in bert_layers[:args.freeze_layers+1]):
                        param.requires_grad = False

        # Shared Layers (with First-Initialized Transformer Encoder)
        else:
            encoder_layer = TransformerEncoderLayer(d_model=args.hidden_size,
                                                    nhead=args.n_heads,
                                                    dim_feedforward=args.pf_dim,
                                                    dropout=args.dropout,
                                                    batch_first=True,
                                                    device=args.device)
            encoder_norm = nn.LayerNorm(normalized_shape=args.hidden_size,
                                        device=args.device)
            self.shared_layers = TransformerEncoder(encoder_layer=encoder_layer,
                                                    num_layers=args.n_enc_layers,
                                                    norm=encoder_norm)
        self.dropout = nn.Dropout(args.dropout)
        
        # Task-Specific Layer for Detection
        self.detect_linear = nn.Linear(args.hidden_size, 2)
        
        # Task-Specific Layers for Correction
        decoder_layer = TransformerDecoderLayer(d_model=args.hidden_size,
                                                nhead=args.n_heads,
                                                dim_feedforward=args.pf_dim,
                                                dropout=args.dropout,
                                                batch_first=True,
                                                device=args.device)
        decoder_norm = nn.LayerNorm(normalized_shape=args.hidden_size,
                                    device=args.device)
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer,
                                          num_layers=args.n_dec_layers,
                                          norm=decoder_norm)
        self.correct_linear = nn.Linear(args.hidden_size, args.vocab_size)


    def get_attn_mask(self, token_ids, valid_length):
        attn_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attn_mask[i][:v] = 1
        return attn_mask.float()
        
        
    def forward(self, src, tgt, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask,
                inference=None):
        # Shared Layers for Text Representations
        if self.with_pretrained:
            src_emb = self.shared_layers(src).last_hidden_state
            # src_emb: [batch_size, max_length, hidden_size] [64, 256, 768]
        else:
            src_emb = self.src_token_embedding(src)
            src_emb = self.pos_embedding(src_emb)
            src_emb = self.shared_layers(src_emb)
        src_emb = self.dropout(src_emb)
        

        # Task-Specific Layer for Detection
        if inference == None:
            out1 = self.detect_linear(src_emb)
        if inference == 'detect':
            return torch.sigmoid(out1)
        if inference == 'correct':
            pass

        
        # Task-Specific Layers for Correction
        if self.with_pretrained:
            tgt_emb = self.shared_layers.embeddings(tgt)
            # tgt_emb: [batch_size, max_length, hidden_size] [64, 256, 768]
        else:
            tgt_emb = self.tgt_token_embedding(tgt)
            tgt_emb = self.pos_embedding(tgt_emb)
    
        out2 = self.decoder(tgt_emb, src_emb,
                            tgt_mask=tgt_mask,
                            memory_mask=None,
                            tgt_key_padding_mask=tgt_pad_mask,
                            memory_key_padding_mask=src_pad_mask)
        # out2: [batch_size, max_length, hidden_size] [64, 256, 768]
        
        out2 = self.correct_linear(out2)
        # out2: [batch_size, max_length, vocab_size] [64, 256, 30000]
        
        return torch.sigmoid(out1), out2
    
    
    def encode(self, src):
        if self.with_pretrained:
            return self.shared_layers(src)
        else:
            return None
    
    
    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.shared_layers.embeddings(tgt)
        return self.decoder(tgt_emb, memory, tgt_mask)
    
    
    def init_decoder(self):
        init_weights(self.decoder)
        
        
def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask


def create_mask(src, tgt, args):
    assert src.shape[1] == args.max_tokens_per_sent
    assert tgt.shape[1] == args.max_tokens_per_sent

    src_seq_len = src.shape[1]
    src_mask    = torch.zeros((src_seq_len, src_seq_len), device=args.device).type(torch.bool)

    tgt_seq_len = tgt.shape[1]
    tgt_mask    = generate_square_subsequent_mask(tgt_seq_len, args.device)

    if args.with_pretrained and 'kobert' in args.model_path:
        pad_idx = 1
    elif args.with_pretrained and 'kcbert' in args.model_path:
        pad_idx = 0
    elif not args.with_pretrained:
        pad_idx = 2

    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def init_weights(model):
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
            
            
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)