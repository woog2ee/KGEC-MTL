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
from trainer_correct_baselines import iteration, predict, predict_batch, predict_beam1
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--data_path', type=str, default='C:/Users/iipl/Desktop/correct_baselines/total_resorted_29_')
    #parser.add_argument('--data_path', type=str, default='/HDD/seunguk/KGEC/total_resorted_29_')
    parser.add_argument('--tokenizer_path', type=str, default='C:/Users/iipl/Desktop/correct_baselines/tokenizer/') 
    #parser.add_argument('--tokenizer_path', type=str, default='/home/seunguk/KGEC/0821/correct_baselines/tokenizer/')
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--vocab_with', type=str, choices=['bpe', 'char'])
    parser.add_argument('--save_path', type=str, default='C:/Users/iipl/Desktop/correct_baselines/')
    #parser.add_argument('--save_path', type=str)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=10)
    
    parser.add_argument('--max_tokens_per_sent', type=int, default=100)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_enc_layers', type=int, default=12)
    parser.add_argument('--n_dec_layers', type=int, default=6)
    parser.add_argument('--pf_dim', type=int, default=768*4)
    
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=1)
    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip', type=int, default=1)
    
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--beam_testbatch_1', type=str2bool, default='false')
    parser.add_argument('--curriculum', type=str2bool, default='false')
    parser.add_argument('--test', type=str2bool, default='true')
    
    
    print('========== Loading All Parse Arguments\n')
    args = parser.parse_args()
    init_seed(args.seed)
    args.vocab_with = 'bpe'
    args.lr = 0.00001
    args.beam_testbatch_1 = 'true'
    
    print('========== Loading Tokenizer\n')
    tokenizer = SPTokenizer(args)

    
    print(f'========== Loading Dataset & DataLoader\n')
    train_data = load_json_data(args.data_path+'train.json')
    valid_data = load_json_data(args.data_path+'valid.json')
    test_data = load_json_data(args.data_path+'test.json')
    if args.test:
        train_data, valid_data, test_data = train_data[:1600], valid_data[:200], test_data[:200]
        
    train_dataset = TransformerDataset(args, train_data, tokenizer)
    valid_dataset = TransformerDataset(args, valid_data, tokenizer)
    test_dataset = TransformerDataset(args, test_data, tokenizer)
    
    train_loader = build_data_loader(train_dataset, args.batch_size, args.num_workers, args.curriculum)
    valid_loader = build_data_loader(valid_dataset, args.batch_size, args.num_workers, args.curriculum)
    if not args.beam_testbatch_1:
        test_loader = build_data_loader(test_dataset, args.batch_size, args.num_workers, args.curriculum)
    else:
        test_loader = build_data_loader(test_dataset, 1, args.num_workers, args.curriculum)
    
    print(f'========== Loading Model\n')
    model = TransformerCorrector(args)
    model.apply(init_weights)
    
   
    print('========== Setting Optimizer & Scheduler\n')
    t_total = len(train_loader) * args.epochs
    warmup_steps = int(t_total) * args.warmup_ratio
    optimizer, scheduler = get_optimizer_and_scheduler(model, args.lr, args.beta1, args.beta2, args.eps, warmup_steps, t_total)
    
    
    print('========== Training & Testing Start\n')
    train_epoch_loss, valid_epoch_loss,\
        train_batch_loss, valid_batch_loss = iteration(args, model, tokenizer.pad_num, train_loader, valid_loader, optimizer, scheduler)
    
    best_epoch = valid_epoch_loss.index(min(valid_epoch_loss))
    #test_metrics = predict(args, tokenizer, best_epoch, test_dataset)
    #test_metrics = predict_batch(args, tokenizer, best_epoch, test_loader)
    test_metrics = predict_beam1(args, tokenizer, best_epoch, test_loader)

    f = open(args.save_path+'_result.txt', 'w')
    print(f'\n========== Best Epoch: {best_epoch+1}, Test Metrics: {test_metrics}')
    f.write(f'\n========== Best Epoch: {best_epoch+1}, Test Metrics: {test_metrics}')

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