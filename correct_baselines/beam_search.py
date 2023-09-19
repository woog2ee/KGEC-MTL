import torch
import torch.nn.functional as F
from model_correct_baselines import generate_square_subsequent_mask


def translate(args, model, tokenizer, sent):
    model.eval()

    src_ids = tokenizer.encode_src(sent)
    src_ids = torch.LongTensor(src_ids).unsqueeze(0).to(args.device)

    num_tokens = src_ids.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(args.device)
    print(f'#### src_ids, mask {src_ids.shape} {src_mask.shape}')
    encoded = model.encode(src_ids, src_mask)
    print(f'#### encoded {encoded.shape}')

    tgt_ids = torch.ones(1, 1).fill_(tokenizer.sos_num).type(torch.long).to(args.device)
    for i in range(int(len(sent)+5)):
        encoded = encoded.to(args.device)
        tgt_mask = (generate_square_subsequent_mask(tgt_ids.size(-1), args.device).type(torch.bool)).to(args.device)

        out = model.decode(tgt_ids, encoded, tgt_mask)
        #print(f'#### out: {out.shape}')
        #out = out.transpose(0, 1)
        prob = model.linear(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        tgt_ids = torch.cat([tgt_ids,
                             torch.ones(1, 1).type_as(src_ids.data).fill_(next_word)], dim=1)
        if next_word == tokenizer.eos_num: break
    print(f'### tgt_ids: {tgt_ids}')
    # tgt_ids: [int(len(sent)*1.5), 1]
    #print(f'tgt_ids: {tgt_ids} {type(tgt_ids)}')
    tgt_ids = tgt_ids.squeeze(0).tolist()
    #print(f'##tgt_ids: {tgt_ids} {type(tgt_ids)}')
    tgt_tokens = tokenizer.decode_tgt(tgt_ids)
    #print(f'###tgt_tokens: {tgt_tokens}')
    #tgt_tokens = [tokenizer.decode_tgt(id) for id in tgt_ids]
    return tgt_tokens

# def translate(args, model, tokenizer, sent):
#     model.eval()

#     src_ids = tokenizer.encode_src(sent)
#     src_ids = torch.LongTensor(src_ids).view(-1, 1).to(args.device)

#     num_tokens = src_ids.shape[0]
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(args.device)
#     encoded = model.encode(src_ids, src_mask)
    
#     tgt_ids = torch.ones(1, 1).fill_(tokenizer.sos_num).type(torch.long).to(args.device)
#     for i in range(int(len(sent)*1.5)):
#         encoded = encoded.to(args.device)
#         tgt_mask = (generate_square_subsequent_mask(tgt_ids.size(0), args.device).type(torch.bool)).to(args.device)

#         out = model.decode(tgt_ids, encoded, tgt_mask)
#         out = out.transpose(0, 1)
#         prob = model.linear(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()

#         tgt_ids = torch.cat([tgt_ids,
#                              torch.ones(1, 1).type_as(src_ids.data).fill_(next_word)], dim=0)
#         if next_word == tokenizer.eos_num: break
#     # tgt_ids: [int(len(sent)*1.5), 1]
#     print(f'tgt_ids: {tgt_ids} {type(tgt_ids)}')
#     tgt_ids = tgt_ids.squeeze(-1).tolist()
#     print(f'##tgt_ids: {tgt_ids} {type(tgt_ids)}')
#     tgt_tokens = tokenizer.decode_tgt(tgt_ids)
#     print(f'###tgt_tokens: {tgt_tokens}')
#     #tgt_tokens = [tokenizer.decode_tgt(id) for id in tgt_ids]
#     return tgt_tokens

def beam_search(args, model, src, src_mask,
                max_len, beam_size, sos_num, eos_num):
    src = src.to(args.device)
    src_mask = src_mask.to(args.device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

    start_token = torch.tensor([sos_num], dtype=torch.long).unsqueeze(0).to(args.device)
    # start_token: [1, 1]
    beam = [(start_token, 0.0, [])]

    for _ in range(max_len):
        
        new_beam = []
        for seq, score, _ in beam:
            tgt_mask = generate_square_subsequent_mask(seq.shape[1], args.device).type(torch.bool).to(args.device)
            # tgt_mask: [seq_len, seq_len]

            with torch.no_grad():
                #print(f'#### seq: {seq.shape}')
                #print(f'#### mem: {memory.shape}')
                #print(f'#### mask: {tgt_mask.shape}\n\n')
                # seq: [1, seq_len]
                # memory: [1, max_seq_len, hidden_size]
                # mask: [seq_len, seq_len]
                decoder_out = model.decode(seq, memory, tgt_mask)
                decoder_out = model.linear(decoder_out)
                # decoder_out: [1, seq_len, vocab_size]

            topk_probs, topk_indices = torch.topk(F.softmax(decoder_out, dim=-1),
                                                 k=beam_size,
                                                 dim=-1)
            # topk_probs, topk_indices: [1, seq_len, beam_size]
            for i in range(beam_size):
                token = topk_indices[0, -1, i].unsqueeze(0).unsqueeze(0)
                #if int(token[0][0]) == int(seq[0][-1]): continue
                # token: [1, 1]

                new_seq = torch.cat([seq, token], dim=1)
                new_score = score + topk_probs[0, -1, i].item()
                #print(f'###{i} {new_seq} {new_score}')
                new_beam.append([new_seq, new_score, []])

        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]

        end_cnt = sum(1 for seq, _, _ in beam if seq[:, -1] == eos_num)
        if end_cnt == beam_size: break
    #return beam
    # Choose the best beam result
    best_seq, best_score, _ = max(beam, key=lambda x: x[1])
    # best_seq: [1, len]
    return best_seq.squeeze(0).tolist()
            


def beam_search_decode(args, model, src, src_mask,
                       max_len, beam_size, start_symbol, end_symbol):
    src = src.to(args.device)
    src_mask = src_mask.to(args.device)
    memory = model.encode(src, src_mask)
    print(f'### memory: {memory.shape}')

    ys = torch.tensor([start_symbol], dtype=torch.long).to(args.device)

    beam_list = [(0.0, ys, False)]  # (log probability, sequence)

    for i in range(max_len - 1):
        new_beam_list = []

        for log_prob, sequence, finish in beam_list:
            tgt_mask = generate_square_subsequent_mask(sequence.size(-1), args.device).type(torch.bool).to(args.device)
            out = model.decode(sequence.unsqueeze(0), memory, tgt_mask)
            #print(f'beam search out : {out.shape}')
            #out = out.transpose(0, 1)
            prob = model.linear(out[:, -1])
            log_probs, next_words = torch.topk(F.log_softmax(prob, dim=1), beam_size, dim=1)

            for j in range(beam_size):
                print(f'next: {next_words[0][j]}')
                new_sequence = torch.cat([sequence, next_words[0][j].unsqueeze(0)], dim=1)
                new_log_prob = log_prob + log_probs[0][j].item()

                if finish:
                    new_beam_list.append((log_prob, sequence, True))
                else:
                    if next_words[0][j].item() == end_symbol:  # EOS
                        new_beam_list.append((log_prob, sequence, True))
                    else:  # not EOS
                        new_beam_list.append((new_log_prob, new_sequence, False))

        new_beam_list.sort(key=lambda x: x[0], reverse=True)
        beam_list = new_beam_list[:beam_size]
    return beam_list[0][1]