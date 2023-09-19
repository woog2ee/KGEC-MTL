import torch
import torch.nn as nn
from model_correct_baselines import create_mask, generate_square_subsequent_mask
from beam_search import translate, beam_search_decode, beam_search
from metrics_correct import compute_correction_metrics_perlist
from tqdm import tqdm


def iteration(args, model, pad_num, train_loader, valid_loader, optimizer, scheduler):
    model = model.to(args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_num).to(args.device)
    
    train_epoch_loss = []
    valid_epoch_loss = []
    train_batch_loss = []
    valid_batch_loss = []
    

    for epoch in range(args.epochs):
        train_epoch_loss_ = 0.0
        valid_epoch_loss_ = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        

        # Train
        train_iter = tqdm(enumerate(train_loader),
                          desc='Epoch_%s:%d' % ('train', epoch+1),
                          total=len(train_loader),
                          bar_format='{l_bar}{r_bar}')
        
        
        model.train()
        for idx, batch in train_iter:
            optimizer.zero_grad()

            #batch = {k: v.to(args.device) for k, v in batch.items()}
            #src, tgt = batch['src'], batch['tgt']
            src = batch[0].to(args.device)
            tgt = batch[1].to(args.device)
            # src, tgt: [batch_size, max_length] [64, 256]
            tgt_input = tgt[:, :-1]

            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input, args)

            out = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask)
            
            tgt = tgt[:, 1:].reshape(-1)
            out = out.reshape(-1, out.shape[-1])
            
            loss = loss_fn(out, tgt)
            train_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            post_fix = {'epoch': epoch+1, 'batch': idx+1,
                        'train_loss': train_loss / (idx+1)}
            if (idx+1) % 100 == 0:
                train_iter.write(str(post_fix))
                train_batch_loss.append(train_loss / (idx+1))

        train_epoch_loss_ = train_loss / (idx+1)
        print(f'===== Epoch {epoch+1} Train Loss: {train_epoch_loss_}')


        # Validation
        valid_iter = tqdm(enumerate(valid_loader),
                        desc='Epoch_%s:%d' % ('valid', epoch+1),
                        total=len(valid_loader),
                        bar_format='{l_bar}{r_bar}')
            
        model.eval()
        for idx, batch in valid_iter:
            #batch = {k: v.to(args.device) for k, v in batch.items()}
            src = batch[0].to(args.device)
            tgt = batch[1].to(args.device)
            # src, tgt: [batch_size, max_length] [64, 256]

            tgt_input = tgt[:, :-1]

            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input, args)
            
            out = model(src, tgt_input, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask)

            tgt = tgt[:, 1:].reshape(-1)
            out = out.reshape(-1, out.shape[-1])
            
            loss = loss_fn(out, tgt)
            valid_loss += loss.item()
        
            post_fix = {'epoch': epoch+1, 'batch': idx+1,
                        'valid_loss': valid_loss / (idx+1)}
            if (idx+1) % 100 == 0:
                valid_iter.write(str(post_fix))
                valid_batch_loss.append(valid_loss / (idx+1))

        valid_epoch_loss_ = valid_loss / (idx+1)
        print(f'===== Epoch {epoch+1} Valid Loss: {valid_epoch_loss_}')


        train_epoch_loss.append(train_epoch_loss_)
        valid_epoch_loss.append(valid_epoch_loss_)

        save_model(model, epoch, args.save_path)

        #scheduler.step()

    return train_epoch_loss, valid_epoch_loss, train_batch_loss, valid_batch_loss

        
def save_model(model, epoch, save_path):
    torch.save(model, save_path+f'_{epoch+1}.pt')
    print(f'===== Epoch {epoch+1} Model Saved at {save_path}_{epoch+1}.pt\n')


def predict_beam1(args, tokenizer, epoch, test_loader):
    # Test
    model = torch.load(args.save_path+f'_{epoch+1}.pt')
    print(f'========== For Testing, {args.save_path}_{epoch+1}.pt Loaded')

    test_iter = tqdm(enumerate(test_loader),
                     desc='Epoch_%s' % ('test'),
                     total=len(test_loader),
                     bar_format='{l_bar}{r_bar}')

    srcs, tgts, pred_tgts = [], [], []
    model.eval()
    for idx, batch in test_iter:
        src = batch[0].to(args.device)
        tgt = batch[1].to(args.device)
        
        num_tokens = src.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        sos_num = tokenizer.sos_num
        eos_num = tokenizer.eos_num
        #sos_num = tokenizer('[CLS]', add_special_tokens=False)['input_ids'][0]
        #eos_num = tokenizer('[SEP]', add_special_tokens=False)['input_ids'][0]

        tgt_tokens = beam_search(args, model, src, src_mask,
                                        max_len=num_tokens+5,
                                        beam_size=args.beam_size,
                                        sos_num=sos_num,
                                        eos_num=eos_num)
        
        #print(f'####### tgt_tokens: {tgt_tokens}')

        src = tokenizer.decode_src(src.squeeze(0).tolist())
        tgt = tokenizer.decode_tgt(tgt.squeeze(0).tolist())
        pred = tokenizer.decode_tgt(tgt_tokens)
        print(f'\n## src         : {src}')
        print(f'### tgt (answer): {tgt}')
        print(f'### pred        : {pred}\n')
        #src_tokens = src.squeeze(0).tolist()

        srcs.append(src)
        tgts.append(tgt)
        pred_tgts.append(pred)
        
        #origin_sent = tokenizer.decode_src(src_tokens)
        #predict_sent = tokenizer.decode_tgt(tgt_tokens)
        #print(f'beam1 src: {src}\n')
        #print(f'beam1 tgt: {tgt_tokens}\n') # list

        #print(f'origin_sent: {origin_sent}\n')
        #print(f'predict_sent: {predict_sent}\n')
    with open(f'{args.save_path}_beam_src.txt', 'w+') as f:
        f.write('\n'.join(srcs))
    with open(f'{args.save_path}_beam_tgt.txt', 'w+') as f:
        f.write('\n'.join(tgts))
    with open(f'{args.save_path}_beam_pred.txt', 'w+') as f:
        f.write('\n'.join(pred_tgts))
        
    
    test_word_prec, test_word_rec, test_word_f1 = compute_correction_metrics_perlist(pred_tgts, tgts, 'word')
    test_char_prec, test_char_rec, test_char_f1 = compute_correction_metrics_perlist(pred_tgts, tgts, 'char')
    return [test_word_prec, test_word_rec, test_word_f1], [test_char_prec, test_char_rec, test_char_f1]



def predict(args, tokenizer, epoch, test_dataset):
    # Test
    model = torch.load(args.save_path+f'_{epoch+1}.pt')
    print(f'========== For Testing, {args.save_path}_{epoch+1}.pt Loaded')

    tgts, pred_tgts = [], []
    model.eval()
    for idx in tqdm(range(len(test_dataset))):
        src, tgt = test_dataset[idx][0], test_dataset[idx][1]
        print(f'src: {src}')
        print(f'tgt: {tgt}\n')

        src = tokenizer.decode_src(src)
        tgt = tokenizer.decode_tgt(tgt)
        #tgts.append(' '.join(tgt))
        print(f'src: {src}')
        print(f'tgt (answer): {tgt}\n')

        pred_tgt = translate(args, model, tokenizer, src)
        #pred_tgt = pred_tgt[1:-1]
        print(f'predict: {pred_tgt}\n\n')

        tgts.append(tgt)
        pred_tgts.append(' '.join(pred_tgt))
        exit()
        #if (idx+1) % 100 == 0:
        #print(f'{idx+1} / {len(test_dataset)}')
        #print(f'answer: {tgt}')
        #print(f'predict: {pred_tgt}\n')

    with open(f'{args.save_path}_greedy_tgt.txt', 'w+') as f:
        f.write('\n'.join(tgts))
    with open(f'{args.save_path}_greedy_pred.txt', 'w+') as f:
        f.write('\n'.join(pred_tgts))

    test_word_prec, test_word_rec, test_word_f1 = compute_correction_metrics_perlist(pred_tgts, tgts, 'word')
    test_char_prec, test_char_rec, test_char_f1 = compute_correction_metrics_perlist(pred_tgts, tgts, 'char')
    return [test_word_prec, test_word_rec, test_word_f1], [test_char_prec, test_char_rec, test_char_f1]