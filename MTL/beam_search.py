import torch
import torch.nn.functional as F
from model_MTL import generate_square_subsequent_mask


def translate(args, model, tokenizer, sent):
    model.eval()

    if args.with_pretrained:
        encoded = model.encode(sent)
        ### 여기 안돌아감
        sos_num = tokenizer(tokenizer.cls_token, add_special_tokens=False)['input_ids'][0]
    else:  
        src_ids = tokenizer.encode_src(sent)
        src_ids = torch.LongTensor(src_ids).view(-1, 1).to(args.device)

        num_tokens = src_ids.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(args.device)
        encoded = model.encode(src_ids, src_mask)

        sos_num = tokenizer.sos_num

    
    tgt_ids = torch.ones(1, 1).fill_(sos_num).type(torch.long).to(args.device)
    for i in range(int(len(sent)*1.5)):
        encoded = encoded.to(args.device)
        tgt_mask = (generate_square_subsequent_mask(tgt_ids.size(0), args.device).type(torch.bool)).to(args.device)

        out = model.decode(tgt_ids, encoded, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.correct_linear(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        tgt_ids = torch.cat([tgt_ids,
                             torch.ones(1, 1).type_as(src_ids.data).fill_(next_word)], dim=0)
        if next_word == tokenizer.eos_num: break
    # tgt_ids: [int(len(sent)*1.5), 1]
    print(f'tgt_ids: {tgt_ids} {type(tgt_ids)}')
    tgt_ids = tgt_ids.squeeze(-1).tolist()
    print(f'##tgt_ids: {tgt_ids} {type(tgt_ids)}')
    tgt_tokens = tokenizer.decode_tgt(tgt_ids)
    print(f'###tgt_tokens: {tgt_tokens}')
    #tgt_tokens = [tokenizer.decode_tgt(id) for id in tgt_ids]
    return tgt_tokens


def beam_search_decode(args, model, src, src_mask, max_len, beam_size, start_symbol, end_symbol):
    src = src.to(args.device)
    src_mask = src_mask.to(args.device)
    memory = model.encode(src).last_hidden_state  # 32,64,768
    print(f'### test memory: {memory.shape}')

    ys = torch.tensor([start_symbol], dtype=torch.long).to(args.device)

    beam_list = [(0.0, ys, False)]  # (log probability, sequence)

    for i in range(max_len - 1):
        new_beam_list = []

        for log_prob, sequence, finish in beam_list:
            tgt_mask = generate_square_subsequent_mask(sequence.size(0), args.device).type(torch.bool).to(args.device)
    
            print(f'### decode input1: {sequence.shape}')
            print(f'### decode input2: {memory.shape}')
            print(f'### decode input3: {tgt_mask.shape}\n')

            out = model.decode(sequence.unsqueeze(0), memory, tgt_mask)
            print(f'test out: {out.shape}')

            
            #out = out.transpose(0, 1)
            prob = model.correct_linear(out[:, -1])
            print(f'test prob: {prob.shape}')

            log_probs, next_words = torch.topk(F.log_softmax(prob, dim=1), beam_size, dim=1)
            print(next_words)
            for j in range(beam_size):
                new_sequence = torch.cat([sequence, next_words[0][j].unsqueeze(0)], dim=0)
                new_log_prob = log_prob + log_probs[0][j].item()

                if finish:
                    new_beam_list.append((log_prob, sequence, True))
                else:
                    if next_words[0][j].item() == end_symbol:  # EOS
                        new_beam_list.append((log_prob, sequence, True))
                    else:  # not EOS
                        print(new_log_prob, new_sequence)
                        new_beam_list.append((new_log_prob, new_sequence, False))

        new_beam_list.sort(key=lambda x: x[0], reverse=True)
        beam_list = new_beam_list[:beam_size]
    return beam_list[0][1]