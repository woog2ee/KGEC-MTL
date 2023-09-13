import torch
import torch.nn as nn
from model_MTL import create_mask, generate_square_subsequent_mask
from beam_search import beam_search_decode, translate
from metrics_detect import compute_metrics
from metrics_correct import compute_correction_metrics_perlist
from tqdm import tqdm


def iteration(args, model, pad_num, train_loader, valid_loader, optimizer, scheduler):
    if args.dataparallel:
        model = nn.DataParallel(model).to(args.device)
    else: 
        model = model.to(args.device)
    loss_fn_detect = nn.CrossEntropyLoss(ignore_index=-1).to(args.device)
    loss_fn_correct = nn.CrossEntropyLoss(ignore_index=pad_num).to(args.device)
    
    train_epoch_loss = []
    valid_epoch_loss = []
    train_batch_loss = []
    valid_batch_loss = []
    
    
    for epoch in range(args.epochs):
        train_epoch_loss_ = 0.0
        valid_epoch_loss_ = 0.0
        train_loss = 0.0
        valid_loss = 0.0
        
    
        train_iter = tqdm(enumerate(train_loader),
                          desc='Epoch_%s:%d' % ('train', epoch+1),
                          total=len(train_loader),
                          bar_format='{l_bar}{r_bar}')
        
        model.train()
        for idx, batch in train_iter:
            optimizer.zero_grad()
            batch = {k: v.to(args.device) for k, v in batch.items()}
            src = batch['src']
            label = batch['label']
            tgt = batch['tgt']
            # src, tgt: [batch_size, max_length] [64, 256]

            # detect & correct 
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt, args)
            # mask: [max_length, max_length] [256, 256]
            # pad_mask: [batch_size, max_length] [64, 256]
            if args.dataparallel:
                src_mask = src_mask.repeat(2, 1)
                tgt_mask = tgt_mask.repeat(2, 1)

            out1, out2 = model(src, tgt, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
            # out2: [max_length, batch_size, vocab_size] [256, 64, 8002]

            out1, label = out1.view(-1, 2), label.view(-1)
            loss_detect = loss_fn_detect(out1, label)

            tgt = tgt.reshape(-1) 
            out2 = out2.reshape(-1, out2.shape[-1])
            loss_correct = loss_fn_correct(out2, tgt)
            
            
            # MTL
            loss = loss_detect + loss_correct
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
            batch = {k: v.to(args.device) for k, v in batch.items()}
            src = batch['src']
            label = batch['label']
            tgt = batch['tgt']
            
            # detect & correct 
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt, args)
            # mask: [max_length, max_length] [256, 256]
            # pad_mask: [batch_size, max_length] [64, 256]

            out1, out2 = model(src, tgt, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask)
            # out2: [max_length, batch_size, vocab_size] [256, 64, 8002]

            out1, label = out1.view(-1, 2), label.view(-1)
            loss_detect = loss_fn_detect(out1, label)
            
            tgt = tgt.reshape(-1)
            out2 = out2.reshape(-1, out2.shape[-1])
            loss_correct = loss_fn_correct(out2, tgt)
            
            
            # MTL
            loss = loss_detect + loss_correct
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
        
    return train_epoch_loss, valid_epoch_loss, train_batch_loss, valid_batch_loss


def save_model(model, epoch, save_path):
    torch.save(model, save_path+f'_{epoch+1}.pt')
    print(f'===== Epoch {epoch+1} Model Saved at {save_path}_{epoch+1}.pt\n')
    
    
def predict_detection(args, epoch, test_loader):
    # Test
    model = torch.load(args.save_path+f'_{epoch+1}.pt')
    test_acc, test_prec, test_rec, test_f1 = 0.0, 0.0, 0.0, 0.0
    print(f'========== For Testing, {args.save_path}_{epoch+1}.pt Loaded')

    test_iter = tqdm(enumerate(test_loader),
                     desc='Epoch_%s' % ('test'),
                     total=len(test_loader),
                     bar_format='{l_bar}{r_bar}')

    model.eval()
    for idx, batch in test_iter:    
        batch = {k: v.to(args.device) for k, v in batch.items()}
        src = batch['src']
        label = batch['label']
        #tgt = batch['tgt']

        # detect
        out1 = model(src, None, None, None, None, None, inference='detect')
            
        out1, label = out1.view(-1, 2), label.view(-1)
        
        out1 = torch.argmax(out1, dim=1)

        acc, prec, rec, f1 = compute_metrics(label, out1)
        test_acc += acc
        test_prec += prec
        test_rec += rec
        test_f1 += f1
    
    test_acc = test_acc / (idx+1)
    test_prec = test_prec / (idx+1)
    test_rec = test_rec / (idx+1)
    test_f1 = test_f1 / (idx+1)
    return [test_acc, test_prec, test_rec, test_f1]

    
def predict_correction(args, tokenizer, epoch, test_dataset):
    # Test
    model = torch.load(args.save_path+f'_{epoch+1}.pt')
    print(f'========== For Testing, {args.save_path}_{epoch+1}.pt Loaded')

    tgts, pred_tgts = [], []
    model.eval()
    for idx in tqdm(range(len(test_dataset))):
        src, tgt = test_dataset[idx][0], test_dataset[idx][1]
        if args.with_pretrained:
            src = tokenizer.decode(src)
            tgt = tokenizer.decode(tgt).split(' ')
        else:
            src = tokenizer.decode_src(src)
            tgt = tokenizer.decode_tgt(tgt).split(' ')
        tgts.append(' '.join(tgt))
        print(f'src: {src}')
        print(f'tgt: {tgt}')

        pred_tgt = translate(args, model, tokenizer, src)
        pred_tgt = pred_tgt[1:-1]
        pred_tgts.append(' '.join(pred_tgt))

        print(f'{idx+1} / {len(test_dataset)}')
        print(f'answer: {tgt}')
        print(f'predict: {pred_tgt}\n')
    
    test_word_prec, test_word_rec, test_word_f1 = compute_correction_metrics_perlist(pred_tgts, tgts, 'word')
    test_char_prec, test_char_rec, test_char_f1 = compute_correction_metrics_perlist(pred_tgts, tgts, 'char')
    return [test_word_prec, test_word_rec, test_word_f1], [test_char_prec, test_char_rec, test_char_f1]