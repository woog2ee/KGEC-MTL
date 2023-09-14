import sentencepiece as spm


class SPTokenizer:
    sos_num = 0
    eos_num = 1
    pad_num = 2
    unk_num = 3
    
    def __init__(self, args):
        self.src_tokenizer = self.init_tokenizer(args.tokenizer_path+args.vocab_with+'_src.model')
        self.tgt_tokenizer = self.init_tokenizer(args.tokenizer_path+args.vocab_with+'_tgt.model')
        
    def init_tokenizer(self, tokenizer_dir):
        tokenizer = spm.SentencePieceProcessor(tokenizer_dir)
        #tokenizer.SetEncodeExtraOptions('bos:eos')
        return tokenizer
    
    def encode_src(self, sent):
        return self.src_tokenizer.EncodeAsIds(sent)
         
    def decode_src(self, ids):
        return self.src_tokenizer.DecodeIds(ids) 
        
    def encode_tgt(self, sent):
        return self.tgt_tokenizer.EncodeAsIds(sent)  
        
    def decode_tgt(self, ids):
        return self.tgt_tokenizer.DecodeIds(ids)