import torch
import torch.nn.functional as F
from model_correct_baselines import generate_square_subsequent_mask


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