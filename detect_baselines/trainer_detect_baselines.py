import torch
import torch.nn as nn
from metrics import compute_metrics
from tqdm import tqdm


def iteration(args, model, train_loader, valid_loader, optimizer, scheduler):
    model = nn.DataParallel(model).to(args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1).to(args.device)
    
    train_epoch_metrics, train_epoch_loss = [], []
    valid_epoch_metrics, valid_epoch_loss = [], []
    train_batch_metrics, train_batch_loss = [], []
    valid_batch_metrics, valid_batch_loss = [], []
    

    for epoch in range(args.epochs):
        train_epoch_acc, train_epoch_prec, train_epoch_rec, train_epoch_f1, train_epoch_loss_ = 0.0, 0.0, 0.0, 0.0, 0.0
        valid_epoch_acc, valid_epoch_prec, valid_epoch_rec, valid_epoch_f1, valid_epoch_loss_ = 0.0, 0.0, 0.0, 0.0, 0.0
        train_acc, train_prec, train_rec, train_f1, train_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        valid_acc, valid_prec, valid_rec, valid_f1, valid_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        

        # Train
        train_iter = tqdm(enumerate(train_loader),
                          desc='Epoch_%s:%d' % ('train', epoch+1),
                          total=len(train_loader),
                          bar_format='{l_bar}{r_bar}')
        
        model.train()
        for idx, batch in train_iter:
            optimizer.zero_grad()
            
            batch = {k: v.to(args.device) for k, v in batch.items()}
            input_ids = batch['input_ids']
            valid_length = batch['valid_length']
            segment_ids = batch['segment_ids']
            label = batch['label']
            # input_ids, label: [batch_size, max_tokens_per_sent]

            out = model(input_ids, valid_length, segment_ids)
            # out: [batch_size, max_tokens_per_sent, 2]
    
            out, label = out.view(-1, 2), label.view(-1)
            # out: [batch_size * max_tokens_per_sent, 2]
            # label: [batch_size * max_tokens_per_sent]
            
            loss = loss_fn(out, label)
            train_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            out = torch.argmax(out, dim=1)
            # out: [batch_size * max_tokens_per_sent]

            acc, prec, rec, f1 = compute_metrics(label, out)
            train_acc += acc
            train_prec += prec
            train_rec += rec
            train_f1 += f1
            
            post_fix = {'epoch': epoch+1, 'batch': idx+1,
                        'train_loss': train_loss / (idx+1),
                        'train_acc': train_acc / (idx+1),
                        'train_prec': train_prec / (idx+1),
                        'train_rec': train_rec / (idx+1),
                        'train_f1': train_f1 / (idx+1)}
            if (idx+1) % 100 == 0:
                train_iter.write(str(post_fix))
                train_batch_metrics.append([train_acc / (idx+1), train_prec / (idx+1),\
                                            train_rec / (idx+1), train_f1 / (idx+1)])
                train_batch_loss.append(train_loss / (idx+1))
                
        train_epoch_acc = train_acc / (idx+1)
        train_epoch_prec = train_prec / (idx+1)
        train_epoch_rec = train_rec / (idx+1)
        train_epoch_f1 = train_f1 / (idx+1)
        train_epoch_loss_ = train_loss / (idx+1)
        print(f'===== Epoch {epoch+1} Train Accuracy: {train_epoch_acc}, Precision: {train_epoch_prec}')
        print(f'===== Epoch {epoch+1} Train Recall: {train_epoch_rec}, F1-score: {train_epoch_f1}')
        print(f'===== Epoch {epoch+1} Train Loss: {train_epoch_loss_}')
        
                                                                 
        # Validation
        valid_iter = tqdm(enumerate(valid_loader),
                          desc='Epoch_%s:%d' % ('valid', epoch+1),
                          total=len(valid_loader),
                          bar_format='{l_bar}{r_bar}')
        
        model.eval()
        for idx, batch in valid_iter:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            input_ids = batch['input_ids']
            valid_length = batch['valid_length']
            segment_ids = batch['segment_ids']
            label = batch['label']
            # input_ids, label: [batch_size, max_tokens_per_sent]

            out = model(input_ids, valid_length, segment_ids)
            # out: [batch_size, max_tokens_per_sent, 2]
            
            out, label = out.view(-1, 2), label.view(-1)
            # out: [batch_size * max_tokens_per_sent, 2]
            # label: [batch_size * max_tokens_per_sent]
            
            loss = loss_fn(out, label)
            valid_loss += loss.item()

            out = torch.argmax(out, dim=1)
            # out: [batch_size * max_tokens_per_sent]

            acc, prec, rec, f1 = compute_metrics(label, out)
            valid_acc += acc
            valid_prec += prec
            valid_rec += rec
            valid_f1 += f1
            
            post_fix = {'epoch': epoch+1, 'batch': idx+1,
                        'valid_loss': valid_loss / (idx+1),
                        'valid_acc': valid_acc / (idx+1),
                        'valid_prec': valid_prec / (idx+1),
                        'valid_rec': valid_rec / (idx+1),
                        'valid_f1': valid_f1 / (idx+1)}
            if (idx+1) % 100 == 0:
                valid_iter.write(str(post_fix))
                valid_batch_metrics.append([valid_acc / (idx+1), valid_prec / (idx+1),\
                                            valid_rec / (idx+1), valid_f1 / (idx+1)])
                valid_batch_loss.append(valid_loss / (idx+1))
                
        valid_epoch_acc = valid_acc / (idx+1)
        valid_epoch_prec = valid_prec / (idx+1)
        valid_epoch_rec = valid_rec / (idx+1)
        valid_epoch_f1 = valid_f1 / (idx+1)
        valid_epoch_loss_ = valid_loss / (idx+1)
        print(f'===== Epoch {epoch+1} Valid Accuracy: {valid_epoch_acc}, Precision: {valid_epoch_prec}')
        print(f'===== Epoch {epoch+1} Valid Recall: {valid_epoch_rec}, F1-score: {valid_epoch_f1}')
        print(f'===== Epoch {epoch+1} Valid Loss: {valid_epoch_loss_}')
        

        train_epoch_metrics.append([train_epoch_acc, train_epoch_prec, train_epoch_rec, train_epoch_f1])
        train_epoch_loss.append(train_epoch_loss_)
        valid_epoch_metrics.append([train_epoch_acc, train_epoch_prec, train_epoch_rec, train_epoch_f1])
        valid_epoch_loss.append(valid_epoch_loss_)

        save_model(model, epoch, args.save_path)

    return train_epoch_metrics, train_epoch_loss, valid_epoch_metrics, valid_epoch_loss,\
        train_batch_metrics, train_batch_loss, valid_batch_metrics, valid_batch_loss
    

def save_model(model, epoch, save_path):
    torch.save(model, save_path+f'_{epoch+1}.pt')
    print(f'===== Epoch {epoch+1} Model Saved at {save_path}_{epoch+1}.pt\n')


def predict(args, epoch, test_loader):
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
        input_ids = batch['input_ids']
        valid_length = batch['valid_length']
        segment_ids = batch['segment_ids']
        label = batch['label']
        # input_ids, label: [batch_size, max_tokens_per_sent]

        out = model(input_ids, valid_length, segment_ids)
        # out: [batch_size, max_tokens_per_sent, 2]
        
        out, label = out.view(-1, 2), label.view(-1)
        # out: [batch_size * max_tokens_per_sent, 2]
        # label: [batch_size * max_tokens_per_sent]

        out = torch.argmax(out, dim=1)
        # out: [batch_size * max_tokens_per_sent]

        acc, prec, rec, f1 = compute_metrics(label, out)
        test_acc += acc
        test_prec += prec
        test_rec += rec
        test_f1 += f1
    
    test_acc = test_acc / (idx+1)
    test_prec = test_prec / (idx+1)
    test_rec = test_rec / (idx+1)
    test_f1 = test_f1 / (idx+1)
    return [test_acc, test_prec, test_rec, test_f1]