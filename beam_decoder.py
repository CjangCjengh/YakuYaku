import torch


class Beam:
    def __init__(self, size, pad, bos, eos, device=False):

        self.size = size
        self._done = False
        self.PAD = pad
        self.BOS = bos
        self.EOS = eos
        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        # Initialize to [BOS, PAD, PAD ..., PAD]
        self.next_ys = [torch.full((size,), self.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_logprob):
        """Update beam status and check if finished or not."""
        num_words = word_logprob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # in initial case,
            beam_lk = word_logprob[0]

        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = torch.div(best_scores_id, num_words, rounding_mode='floor')
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """Sort the scores."""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep."""

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        # print(k.type())
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


def beam_search(model, src, src_mask, max_len, pad, bos, eos, beam_size, device, is_terminated=None):
    """ Translation work in one batch """

    def subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.uint8), diagonal=1)
        return subsequent_mask == 0

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        """ Indicate the position of an instance in a tensor. """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        """ Collect tensor parts associated to active instances. """

        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        # active instances (elements of batch) * beam search size x seq_len x h_dimension
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        # select only parts of tensor which are still active
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(
            src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_src_mask = collect_active_part(src_mask, active_inst_idx, n_prev_active_inst, beam_size)

        return active_src_enc, active_src_mask, active_inst_idx_to_position_map

    def beam_decode_step(
            inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
        """ Decode and update beam status, and then return active beam idx """

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            # Batch size x Beam size x Dec Seq Len
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            # Batch size*Beam size x Dec Seq Len
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
            assert enc_output.shape[0] == dec_seq.shape[0] == src_mask.shape[0]
            out = model.decode(enc_output, src_mask,
                               dec_seq,
                               subsequent_mask(dec_seq.size(1))
                               .type_as(src.data))
            word_logprob = model.generator(out[:, -1])
            word_logprob = word_logprob.view(n_active_inst, n_bm, -1)

            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(
                    word_prob[inst_position])  # Fill Beam object with assigned probabilities
                if not is_inst_complete:  # if top beam ended with eos, we do not add it
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)

        # get decoding sequence for each beam
        # size: Batch size*Beam size x Dec Seq Len
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)

        # get word probabilities for each beam
        # size: Batch size x Beam size x Vocabulary
        word_logprob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_logprob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        # -- Encode
        src_enc = model.encode(src, src_mask)

        #  Repeat data for beam search
        NBEST = beam_size
        batch_size, sent_len, h_dim = src_enc.size()
        src_enc = src_enc.repeat(1, beam_size, 1).view(batch_size * beam_size, sent_len, h_dim)
        src_mask = src_mask.repeat(1, beam_size, 1).view(batch_size * beam_size, 1, src_mask.shape[-1])

        # -- Prepare beams
        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device) for _ in range(batch_size)]

        # -- Bookkeeping for active or not
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        # -- Decode
        for len_dec_seq in range(1, max_len + 1):

            if is_terminated is not None and is_terminated():
                return None, None

            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, beam_size)

            if not active_inst_idx_list:
                break  # all instances have finished their path to <EOS>
            # filter out inactive tensor parts (for already decoded sequences)
            src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, NBEST)

    return batch_hyp, batch_scores
