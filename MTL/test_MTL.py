import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from utils_MTL import (str2bool,
                       init_seed,
                       load_json_data,
                       get_pretrained_tokenizer,
                       get_pretrained_model,
                       get_optimizer_and_scheduler)
from tokenizer_correct_baselines import SPTokenizer
from dataset_MTL import MTLDataset
from model_MTL import MultiTaskLearner
from trainer_MTL import iteration, predict_detection, predict_correction, predict_beam1
import os
import re, ast
import argparse


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disabling parallelism to avoid deadlocks
    torch.multiprocessing.set_start_method('spawn')  # to use CUDA with multiprocessing

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--data_path', type=str, default='/HDD/seunguk/KGECdataset/kgec_kowiki_0907')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--with_pretrained', type=str2bool, default='true')
    #parser.add_argument('--tokenizer_path', type=str, default='/home/seunguk/KGEC/0821/correct_baselines/tokenizer/')
    parser.add_argument('--vocab_size', type=int, default=30000)
    #parser.add_argument('--vocab_with', type=str, choices=['bpe', 'char'])
    parser.add_argument('--save_path', type=str)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--max_tokens_per_sent', type=int, default=64)
    

    #parser.add_argument('--batch_size', type=int)
    parser.add_argument('--beam_size', type=int)
    

    print('========== Loading All Parse Arguments\n')
    args = parser.parse_args()
    init_seed(args.seed)

    
    print('========== Loading Tokenizer\n')
    #tokenizer = SPTokenizer(args)
    tokenizer = get_pretrained_tokenizer(args.model_path)
    pad_num = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
    
    args.vocab_size = tokenizer.vocab_size


    # get best epoch
    def only_text(text):
        text = re.sub(r'[=:\n.,]+', '', text)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[\[\]]', '', text)
        return text.strip()

    with open(args.save_path+'_result.txt', 'r') as f:
        result = f.readlines()
    result_ = [only_text(t) for t in result]
    valid_epoch_loss = result[result_.index('Valid Epoch Loss')]

    def preprocess_result(text):
        text = re.sub(r'[a-zA-Z]+', '', text)
        text = re.sub(r'[=:\n]+', '', text)
        text = text.strip()
        
        array = ast.literal_eval(text)
        return array
    valid_epoch_loss = preprocess_result(valid_epoch_loss)
    best_epoch = valid_epoch_loss.index(min(valid_epoch_loss))


    print(f'========== Loading Test Dataset & DataLoader\n')
    test_data = load_json_data(args.data_path+'test.json')
    test_dataset = MTLDataset(args, test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             sampler=RandomSampler(test_dataset), num_workers=args.num_workers)
    
    print(f'beam size: {args.beam_size}')
    #test_word_metrics, test_char_metrics = predict(args, tokenizer, best_epoch, test_dataset)
    predict_beam1(args, tokenizer, best_epoch, test_loader)