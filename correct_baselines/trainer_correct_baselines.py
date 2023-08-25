import torch
import torch.nn as nn
from model_correct_baselines import create_mask, generate_square_subsequent_mask
from beam_search import beam_search_decode
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


def translate(args, model, tokenizer, sent):
    model.eval()

    src_ids = tokenizer.encode_src(sent)
    src_ids = torch.LongTensor(src_ids).view(-1, 1).to(args.device)

    num_tokens = src_ids.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(args.device)
    encoded = model.encode(src_ids, src_mask)

    tgt_ids = torch.ones(1, 1).fill_(tokenizer.sos_num).type(torch.long).to(args.device)
    for i in range(int(len(sent)*1.5)):
        encoded = encoded.to(args.device)
        tgt_mask = (generate_square_subsequent_mask(tgt_ids.size(0), args.device).type(torch.bool)).to(args.device)

        out = model.decode(tgt_ids, encoded, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.linear(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        tgt_ids = torch.cat([tgt_ids,
                             torch.ones(1, 1).type_as(src_ids.data).fill_(next_word)], dim=0)
        if next_word == tokenizer.eos_num: break
    # tgt_ids: [int(len(sent)*1.5), 1]
    print(f'tgt_ids: {tgt_ids}')
    print(f'tgt_ids.shape: {tgt_ids.shape}')
    tgt_ids = tgt_ids.squeeze(-1).tolist()
    tgt_tokens = [tokenizer.decode_tgt(id) for id in tgt_ids]
    print(f'tgt_tokens: {tgt_tokens}')
    return tgt_tokens


def predict_beam1(args, tokenizer, epoch, test_loader):
    # Test
    model = torch.load(args.save_path+f'_{epoch+1}.pt')
    test_acc, test_prec, test_rec, test_f1, test_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    print(f'========== For Testing, {args.save_path}_{epoch+1}.pt Loaded')

    test_iter = tqdm(enumerate(test_loader),
                     desc='Epoch_%s' % ('test'),
                     total=len(test_loader),
                     bar_format='{l_bar}{r_bar}')

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



def predict_batch(args, tokenizer, epoch, test_loader):
    # Test
    model = torch.load(args.save_path+f'_{epoch+1}.pt')
    test_acc, test_prec, test_rec, test_f1, test_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    print(f'========== For Testing, {args.save_path}_{epoch+1}.pt Loaded')

    test_iter = tqdm(enumerate(test_loader),
                     desc='Epoch_%s' % ('test'),
                     total=len(test_loader),
                     bar_format='{l_bar}{r_bar}')

    model.eval()
    for idx, batch in test_iter:
        src, tgt = batch[0].T, batch[1].T
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt[:-1, :], args.device)
        
        out = model(src, tgt[:-1, :], src_mask, tgt_mask,
                    src_pad_mask, tgt_pad_mask, src_pad_mask)
        print(f'batch before tgt: {tgt.shape}')
        print(f'batch before out: {out.shape}')

        tgt = tgt[1:, :].reshape(-1)
        out = out.reshape(-1, out.shape[-1])
        print(f'batch tgt: {tgt.shape}')
        print(f'batch out: {out.shape}')
        exit()
        #loss = loss_fn(out, tgt)
        test_loss += loss.item()



def predict(args, tokenizer, epoch, test_dataset):
    # Test
    model = torch.load(args.save_path+f'_{epoch+1}.pt')
    test_acc, test_prec, test_rec, test_f1 = 0.0, 0.0, 0.0, 0.0
    print(f'========== For Testing, {args.save_path}_{epoch+1}.pt Loaded')

    tgts, pred_tgts = [], []
    model.eval()
    for idx in tqdm(range(len(test_dataset))):
        src, tgt = test_dataset[idx][0], test_dataset[idx][1]
        src = tokenizer.decode_src(src)
        tgt = tokenizer.decode_tgt(tgt).split(' ')
        tgts.append([tgt])
        print(f'src: {src}')
        print(f'tgt: {tgt}')

        pred_tgt = translate(args, model, tokenizer, src)
        pred_tgt = pred_tgt[1:-1]
        pred_tgts.append(pred_tgt)

        print(f'{idx+1} / {len(test_dataset)}')
        print(f'answer: {tgt}')
        print(f'predict: {pred_tgt}\n')
        
        # compute_metrics KAGAS
        acc, prec, rec, f1 = 1,2,3,4#compute_metrics(label, out)
        test_acc += acc
        test_prec += prec
        test_rec += rec
        test_f1 += f1

    test_acc = test_acc / (idx+1)
    test_prec = test_prec / (idx+1)
    test_rec = test_rec / (idx+1)
    test_f1 = test_f1 / (idx+1)
    return [test_acc, test_prec, test_rec, test_f1]