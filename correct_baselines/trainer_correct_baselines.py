import torch
import torch.nn as nn
from model_correct_baselines import create_mask, generate_square_subsequent_mask
from beam_search import translate, beam_search_decode
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
            
            src, tgt = batch[0].T, batch[1].T
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt[:-1, :], args.device)
            
            out = model(src, tgt[:-1, :], src_mask, tgt_mask,
                        src_pad_mask, tgt_pad_mask, src_pad_mask)
            
            tgt = tgt[1:, :].reshape(-1)
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
            src, tgt = batch[0].T, batch[1].T
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt[:-1, :], args.device)
            
            out = model(src, tgt[:-1, :], src_mask, tgt_mask,
                        src_pad_mask, tgt_pad_mask, src_pad_mask)
            
            tgt = tgt[1:, :].reshape(-1)
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

    tgts, pred_tgts = [], []
    model.eval()
    for idx, batch in test_iter:
        src, tgt = batch[0].T, batch[1].T

        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        tgt_tokens = beam_search_decode(args, model, src, src_mask,
                                        max_len=int(num_tokens * 1.5),
                                        beam_size=args.beam_size,
                                        start_symbol=tokenizer.sos_num)
        tgt_tokens = tgt_tokens.tolist()

        src_tokens = src.squeeze(-1).tolist()

        origin_sent = tokenizer.decode_src(src_tokens)
        predict_sent = tokenizer.decode_tgt(tgt_tokens)
        print(f'beam1 src: {src}\n')
        print(f'beam1 tgt: {tgt_tokens}\n') # list

        print(f'origin_sent: {origin_sent}\n')
        print(f'predict_sent: {predict_sent}\n')
        exit()
    
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