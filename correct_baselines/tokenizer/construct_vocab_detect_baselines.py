import sentencepiece as spm
import os
import json
import argparse
from tqdm import tqdm


def json2txt(args):
    json_path = os.path.join(args.data_path, args.data_name+'.json')
    txt_path = os.path.join(args.data_path, args.data_name+args.vocab_with+'.txt')
    
    with open(json_path, 'r') as json_f, open(txt_path, 'w') as txt_f:
        dataset = json.load(json_f)
        
        for item in tqdm(dataset):
            assert len(item) == 3  # item is consists of [src, tgt, difficulty]

            if args.vocab_with == 'src':
                txt_f.write(item[0] + '\n')
            elif args.vocab_with == 'tgt':
                txt_f.write(item[1] + '\n')
            elif args.vocab_with == 'both':
                txt_f.write(item[0] + '\n' + item[1] + '\n')
            else:
                raise Exception('wrong --vocab_with argument detected')
    return True


def construct_sp_vocab(args):
    templates = '--input={}\
                 --model_prefix={} --vocab_size={} --character_coverage={} --model_type={}\
                 --bos_id=0 --eos_id=1 --pad_id=2 --unk_id=3\
                 --bos_piece=<s> --eos_piece=</s> --pad_piece=<pad> --unk_piece=<unk>'
    txt_path = os.path.join(args.data_path, args.data_name+args.vocab_with+'.txt')

    # File naming with bpe_src.vocab, bpe_tgt.vocab, char_src.vocab, char_tgt.vocab
    cmd = templates.format(txt_path, args.prefix+args.model_type+f'_{args.vocab_with}',
                           args.vocab_size, args.char_coverage, args.model_type)
    spm.SentencePieceTrainer.train(cmd)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='/HDD/seunguk/KGEC')
    parser.add_argument('--data_name', type=str, default='total_resorted_29_test')
    parser.add_argument('--vocab_with', type=str, choices=['src', 'tgt', 'both'])
    
    parser.add_argument('--prefix', type=str, default='/home/seunguk/KGEC/0821/correct_baselines/tokenizer/')
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--char_coverage', type=float, default=0.9999)
    parser.add_argument('--model_type', type=str, choices=['bpe', 'char'])
    
    args = parser.parse_args([])
    
    # Change options
    args.vocab_with = 'tgt'
    args.char_coverage = 0.9999
    args.model_type = 'bpe'
    

    # Convert json file to text file
    txt_path = os.path.join(args.data_path, args.data_name+args.vocab_with+'.txt')
    if os.path.exists(txt_path):
        os.remove(txt_path)

    json2txt(args)
    print('text file created\n')
        

    # Make vocabulary
    vocab_path = os.path.join(args.prefix, args.model_type+f'_{args.vocab_with}')
    if os.path.exists(vocab_path+'.model'):
        os.remove(vocab_path+'.model')
        os.remove(vocab_path+'.vocab')
    
    construct_sp_vocab(args)
    print('vocabulary created\n')