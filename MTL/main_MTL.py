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
from trainer_MTL import iteration, predict_detection, predict_correction
import os
import argparse


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disabling parallelism to avoid deadlocks
    torch.multiprocessing.set_start_method('spawn')  # to use CUDA with multiprocessing

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataparallel', type=str2bool, default='false')
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--data_path', type=str, default='/HDD/seunguk/KGECdataset/kgec_kowiki_')
    parser.add_argument('--model_path', type=str, choices=['monologg/kobert', 'beomi/kcbert-base'])
    parser.add_argument('--with_pretrained', type=str2bool, default='true')
    
    parser.add_argument('--tokenizer_path', type=str, default='/home/seunguk/KGEC/0821/correct_baselines/tokenizer/')
    parser.add_argument('--vocab_size', type=int, default=30000)
    parser.add_argument('--vocab_with', type=str, choices=['bpe', 'char'])
    parser.add_argument('--save_path', type=str)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=10)
    
    parser.add_argument('--max_tokens_per_sent', type=int, default=256)
    parser.add_argument('--freeze_layers', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_enc_layers', type=int, default=12)
    parser.add_argument('--n_dec_layers', type=int, default=6)
    parser.add_argument('--pf_dim', type=int, default=768*4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip', type=int, default=5)
    
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--beam_testbatch_1', type=str2bool, default='false')
    parser.add_argument('--curriculum', type=str2bool, default='false')
    parser.add_argument('--test', type=str2bool, default='true')
    
    
    print('========== Loading All Parse Arguments\n')
    args = parser.parse_args()
    init_seed(args.seed)
    print(args.dataparallel)
    
    print('========== Loading Tokenizer\n')
    if args.with_pretrained:
        tokenizer = get_pretrained_tokenizer(args.model_path)
        pad_num = tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        args.vocab_size = tokenizer.vocab_size
    else:
        tokenizer = SPTokenizer(args)
        pad_num = 2
        
        
    print('========== Loading Model\n')
    if args.with_pretrained:
        pretrained_model = get_pretrained_model(args.model_path)
    else:
        pretrained_model = None

    model = MultiTaskLearner(args, pretrained_model)
    model.init_decoder()
    
    
    print('========== Loading Dataset & DataLoader\n')
    train_data = load_json_data(args.data_path+'train.json')
    valid_data = load_json_data(args.data_path+'valid.json')
    test_data = load_json_data(args.data_path+'test.json')
    if args.test:
        train_data, valid_data, test_data = train_data[:800], valid_data[:200], test_data[:200]
    
    train_dataset = MTLDataset(args, train_data, tokenizer)
    valid_dataset = MTLDataset(args, valid_data, tokenizer)
    test_dataset = MTLDataset(args, test_data, tokenizer)

    if args.curriculum:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=SequentialSampler(train_dataset), num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=RandomSampler(train_dataset), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              sampler=RandomSampler(valid_dataset), num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             sampler=RandomSampler(test_dataset), num_workers=args.num_workers)
    
    
    print('========== Setting Optimizer & Scheduler\n')
    t_total = len(train_loader) * args.epochs
    warmup_steps = int(t_total) * args.warmup_ratio
    optimizer, scheduler = get_optimizer_and_scheduler(model, args.lr, args.beta1, args.beta2, args.eps, warmup_steps, t_total)
    
    
    print('========== Training & Testing Start\n')
    train_epoch_loss, valid_epoch_loss,\
        train_batch_loss, valid_batch_loss = iteration(args, model, pad_num,
                                        train_loader, valid_loader, optimizer, scheduler)
    
    best_epoch = valid_epoch_loss.index(min(valid_epoch_loss))
    test_metrics = predict_detection(args, best_epoch, test_loader)
    #test_word_metrics, test_char_metrics = predict_correction(args, tokenizer, best_epoch, test_dataset2)
    # correct 인퍼런스 돌아가게 짜야 함

    f = open(args.save_path+'_result.txt', 'w')
    print(f'\n========== Best Epoch: {best_epoch+1}, Test Metrics: {test_metrics}')
    f.write(f'\n========== Best Epoch: {best_epoch+1}, Test Metrics: {test_metrics}\n')
    #print(f'\n========== Best Epoch: {best_epoch+1}, Detect Metrics: {test_metrics}, Correct Word Metrics: {test_word_metrics}, Char Metrics: {test_char_metrics}')
    #f.write(f'\n========== Best Epoch: {best_epoch+1}, Detect Metrics: {test_metrics}, Correct Word Metrics: {test_word_metrics}, Char Metrics: {test_char_metrics}')

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