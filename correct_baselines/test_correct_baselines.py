import torch
from torch.utils.data import DataLoader, RandomSampler
from utils import (str2bool,
                   init_seed,
                   load_json_data,
                   get_optimizer_and_scheduler)
from tokenizer_correct_baselines import SPTokenizer
from dataset_correct_baselines import TransformerDataset, build_data_loader
from model_correct_baselines import (TransformerCorrector,
                                     init_weights, count_params)
from trainer_correct_baselines import iteration, predict, predict_beam1
import re, ast
import argparse


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # to use CUDA with multiprocessing

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--data_path', type=str, default='/HDD/seunguk/KGECdataset/kgec_kowiki_0907')
    parser.add_argument('--tokenizer_path', type=str, default='/home/seunguk/KGEC/0821/correct_baselines/tokenizer/')
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--vocab_with', type=str, choices=['bpe', 'char'])
    parser.add_argument('--save_path', type=str)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--beam_size', type=int, default=3)
    

    print('========== Loading All Parse Arguments\n')
    args = parser.parse_args()
    init_seed(args.seed)

    
    print('========== Loading Tokenizer\n')
    tokenizer = SPTokenizer(args)


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
    test_dataset = load_json_data(args.data_path+'test.json')
    test_loader = build_data_loader(test_dataset, args, 'test')

    test_word_metrics, test_char_metrics = predict(args, tokenizer, best_epoch, test_dataset)
    predict_beam1(args, tokenizer, best_epoch, test_loader)