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

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--curriculum', type=str2bool, default='false')
    parser.add_argument('--test', type=str2bool, default='false')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--max_tokens_per_sent', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_enc_layers', type=int)
    parser.add_argument('--n_dec_layers', type=int, default=6)
    parser.add_argument('--pf_dim', type=int, default=768*4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip', type=int, default=1)
    
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--beam_testbatch_1', type=str2bool, default='false')
    
    
    print('========== Loading All Parse Arguments\n')
    args = parser.parse_args()
    init_seed(args.seed)

    
    print('========== Loading Tokenizer\n')
    tokenizer = SPTokenizer(args)
    pad_num = tokenizer.pad_num

    
    print(f'========== Loading Dataset & DataLoader\n')
    train_data = load_json_data(args.data_path+'train.json')
    valid_data = load_json_data(args.data_path+'valid.json')
    test_data = load_json_data(args.data_path+'test.json')
    if args.test:
        train_data, valid_data, test_data = train_data[:600], valid_data[:60], test_data[:60]

    train_dataset = TransformerDataset(args, train_data, tokenizer)
    valid_dataset = TransformerDataset(args, valid_data, tokenizer)
    test_dataset = TransformerDataset(args, test_data, tokenizer)
    
    train_loader = build_data_loader(train_dataset, args, 'train')
    valid_loader = build_data_loader(valid_dataset, args, 'valid')
    test_loader = build_data_loader(test_dataset, args, 'test')
    

    print(f'========== Loading Model\n')
    model = TransformerCorrector(args)
    model.apply(init_weights)
    
   
    print('========== Setting Optimizer & Scheduler\n')
    t_total = len(train_loader) * args.epochs
    warmup_steps = int(t_total) * args.warmup_ratio
    optimizer, scheduler = get_optimizer_and_scheduler(model, args.lr, args.beta1, args.beta2, args.eps, warmup_steps, t_total)
    #optimizer, _ = get_optimizer_and_scheduler(model, args.lr, args.beta1, args.beta2, args.eps, warmup_steps, t_total)
    
    
    print('========== Training & Testing Start\n')
    train_epoch_loss, valid_epoch_loss,\
        train_batch_loss, valid_batch_loss = iteration(args, model, pad_num,
                                        train_loader, valid_loader, optimizer, scheduler)
    
    best_epoch = valid_epoch_loss.index(min(valid_epoch_loss))
    #test_word_metrics, test_char_metrics = predict(args, tokenizer, best_epoch, test_dataset)
    # correct 인퍼런스 돌아가게 짜야 함
    #predict_beam1(args, tokenizer, best_epoch, test_loader)
    #test_metrics = predict_batch(args, tokenizer, best_epoch, test_loader)
    #test_metrics = predict_beam1(args, tokenizer, best_epoch, test_loader)

    f = open(args.save_path+'_result.txt', 'w')
    print(f'\n========== Best Epoch: {best_epoch+1}')
    f.write(f'\n========== Best Epoch: {best_epoch+1}')
    # print(f'\n========== Best Epoch: {best_epoch+1}, Test Word Metrics: {test_word_metrics}, Char Metrics: {test_char_metrics}')
    # f.write(f'\n========== Best Epoch: {best_epoch+1}, Test Word Metrics: {test_word_metrics}, Char Metrics: {test_char_metrics}')

    f.write('========== Train Epoch Loss: ')
    f.write(str(train_epoch_loss))
    f.write('\n')
    f.write('========== Valid Epoch Loss: ')
    f.write(str(valid_epoch_loss))
    f.write('\n\n')

    f.write('========== Train Batch Loss: ')
    f.write(str(train_batch_loss))
    f.write('\n')
    f.write('========== Valid Batch Loss: ')
    f.write(str(valid_batch_loss))
    f.write('\n\n')