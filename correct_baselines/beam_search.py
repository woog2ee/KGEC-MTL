import torch
import torch.nn.functional as F
from model_correct_baselines import generate_square_subsequent_mask


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
    tgt_ids = tgt_ids.squeeze(-1).tolist()
    tgt_tokens = [tokenizer.decode_tgt(id) for id in tgt_ids]
    return tgt_tokens


def beam_search_decode(args, model, src, src_mask, max_len, beam_size, start_symbol):
    src = src.to(args.device)
    src_mask = src_mask.to(args.device)
    memory = model.encode(src, src_mask)

    ys = torch.tensor([start_symbol], dtype=torch.long).to(args.device)

    beam_list = [(0.0, ys, False)]  # (log probability, sequence)

    for i in range(max_len - 1):
        new_beam_list = []

        for log_prob, sequence, finish in beam_list:
            tgt_mask = generate_square_subsequent_mask(sequence.size(0), args.device).type(torch.bool).to(args.device)
            out = model.decode(sequence.unsqueeze(1), memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.linear(out[:, -1])
            log_probs, next_words = torch.topk(F.log_softmax(prob, dim=1), beam_size, dim=1)

            for j in range(beam_size):
                new_sequence = torch.cat([sequence, next_words[0][j].unsqueeze(0)], dim=0)
                new_log_prob = log_prob + log_probs[0][j].item()

                if finish:
                    new_beam_list.append((log_prob, sequence, True))
                else:
                    if next_words[0][j].item() == 1:  # EOS
                        new_beam_list.append((log_prob, sequence, True))
                    else:  # not EOS
                        new_beam_list.append((new_log_prob, new_sequence, False))

        new_beam_list.sort(key=lambda x: x[0], reverse=True)
        beam_list = new_beam_list[:beam_size]
    return beam_list[0][1]