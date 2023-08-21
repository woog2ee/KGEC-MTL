import torch
from torch.utils.data import DataLoader, RandomSampler
from utils import (str2bool,
                   init_seed,
                   load_json_data,
                   get_model_and_tokenizer,
                   get_optimizer_and_scheduler)
from dataset_detect_baselines import BERTDataset
from model_detect_baselines import BERTClassifier
from trainer_detect_baselines import iteration, predict
import os
import argparse


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disabling parallelism to avoid deadlocks

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--data_path', type=str, default='/HDD/seunguk/KGEC/total_resorted_29_')
    parser.add_argument('--model_path', type=str,
                        choices=['monologg/kobert', 'monologg/koelectra-base-discriminator',
                                 'beomi/kcbert-base', 'beomi/KcELECTRA-base-v2022'])
    parser.add_argument('--save_path', type=str)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=10)
    
    parser.add_argument('--max_tokens_per_sent', type=int, default=128)
    parser.add_argument('--max_words_per_sent', type=int, default=16)
    parser.add_argument('--detect_level', type=str, default='token', choices=['word', 'token'])
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=768)
    
    parser.add_argument('--linear_num', type=int, default=1, choices=[1, 2])
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip', type=int, default=1)
    
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--curriculum', type=str2bool, default='false')
    parser.add_argument('--test', type=str2bool, default='false')
    
    
    print('========== Loading All Parse Arguments\n')
    args = parser.parse_args()
    init_seed(args.seed)


    print(f'========== Loading Model & Tokenizer with {args.model_path}\n')
    model, tokenizer = get_model_and_tokenizer(args.model_path)
    classifier = BERTClassifier(args, model)
    
    
    print(f'========== Loading Dataset & DataLoader\n')
    train_data = load_json_data(args.data_path+'train.json')
    valid_data = load_json_data(args.data_path+'valid.json')
    test_data = load_json_data(args.data_path+'test.json')
    if args.test:
        train_data, valid_data, test_data = train_data[:1600], valid_data[:200], test_data[:200]
        
    train_dataset = BERTDataset(args, train_data, tokenizer)
    valid_dataset = BERTDataset(args, valid_data, tokenizer)
    test_dataset = BERTDataset(args, test_data, tokenizer)
    
    if args.curriculum:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=RandomSampler(train_dataset), num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  sampler=RandomSampler(valid_dataset), num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 sampler=RandomSampler(test_dataset),num_workers=args.num_workers)
        
    
    print('========== Setting Optimizer & Scheduler\n')
    t_total = len(train_loader) * args.epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    optimizer, scheduler = get_optimizer_and_scheduler(model, args.lr, warmup_steps, t_total)
    
    
    print('========== Training & Testing Start\n')
    train_epoch_metrics, train_epoch_loss, valid_epoch_metrics, valid_epoch_loss,\
        train_batch_metrics, train_batch_loss, valid_batch_metrics, valid_batch_loss = iteration(args, classifier, train_loader, test_loader, optimizer, scheduler)
    
    best_epoch = valid_epoch_loss.index(min(valid_epoch_loss))
    test_metrics = predict(args, best_epoch, test_loader)

    f = open(args.save_path+'_result.txt', 'w')
    print(f'\n========== Best Epoch: {best_epoch+1}, Test Metrics: {test_metrics}')
    f.write(f'\n========== Best Epoch: {best_epoch+1}, Test Metrics: {test_metrics}')

    f.write('========== Train Epoch Metrics: ')
    f.write(str(train_epoch_metrics))
    f.write('\n')
    f.write('========== Valid Epoch Metrics: ')
    f.write(str(valid_epoch_metrics))
    f.write('\n\n')

    f.write('========== Train Epoch Loss: ')
    f.write(str(train_epoch_loss))
    f.write('\n')
    f.write('========== Valid Epoch Loss: ')
    f.write(str(valid_epoch_loss))
    f.write('\n\n')

    f.write('========== Train Batch Metrics: ')
    f.write(str(train_batch_metrics))
    f.write('\n')
    f.write('========== Valid Batch Metrics: ')
    f.write(str(valid_batch_metrics))
    f.write('\n\n')

    f.write('========== Train Batch Loss: ')
    f.write(str(train_batch_loss))
    f.write('\n')
    f.write('========== Valid Batch Loss: ')
    f.write(str(valid_batch_loss))
    f.write('\n\n')