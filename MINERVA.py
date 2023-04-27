from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical

from kb import KB


def ones_var_cuda(s, requires_grad=False):
    return Variable(torch.ones(s), requires_grad=requires_grad).cuda()


def zeros_var_cuda(s, requires_grad=False):
    return Variable(torch.zeros(s), requires_grad=requires_grad).cuda()


def int_fill_var_cuda(s, value, requires_grad=False):
    return int_var_cuda((torch.zeros(s) + value), requires_grad=requires_grad)


def int_var_cuda(x, requires_grad=False):
    return Variable(x, requires_grad=requires_grad).long().cuda()


def var_cuda(x, requires_grad=False):
    return Variable(x, requires_grad=requires_grad).cuda()


EPSILON = float(np.finfo(float).eps)

class StackedRNNCell(nn.Module):
    """
    impl of stacked rnn cell.
    """

    def __init__(self, args, cell_type, in_size, h_size, num_layers=3):
        super(StackedRNNCell, self).__init__()
        self.cells = nn.ModuleList()
        self.args = args
        self.device = torch.device(args.cuda if self.args.gpu else "cpu")
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.rnn = None
        if cell_type == 'lstm':
            self.rnn = nn.LSTMCell
        elif cell_type == 'gru':
            self.rnn = nn.GRUCell
        elif cell_type == 'rnn':
            self.rnn = nn.RNNCell
        if self.rnn is None:
            self.cell_type = 'lstm'
            self.rnn = nn.LSTMCell

        for _ in range(num_layers):
            if args.gpu:
                self.cells.append(self.rnn(in_size, h_size).to(self.device))
            else:
                self.cells.append(self.rnn(in_size, h_size))
            in_size = h_size

    def forward(self, x, hiddens):

        """
        :param x: input embedding
        :param hiddens: an array of length num_layers, hiddens[j] contains (h_t, c_t) of jth layer
        :return:
        """
        input = x
        hiddens_out = []

        for l in range(self.num_layers):
            h_out = self.cells[l](input, hiddens[l])

            hiddens_out.append(h_out)
            input = h_out[0] if self.cell_type == 'lstm' else h_out
        return hiddens_out


class Minerva(nn.Module):
    def __init__(self, args, KB):
        super(Minerva, self).__init__()
        if args.use_entity_embeddings:
            self.e_embeddings = nn.Embedding(args.e_vocab_size, args.embedding_size, padding_idx=0)
            torch.nn.init.xavier_uniform_(self.e_embeddings.weight)
            if not args.train_entity_embeddings:
                self.e_embeddings.requires_grad_(False)
        elif args.use_neighbourhood_embeddings:
            self.neighbourhood_embedding_layer = nn.Linear(args.embedding_size, 1)
            self.r_adjacency = torch.zeros(args.e_vocab_size, args.r_vocab_size)
            # Ensure rows for <NULL> and <NO_OP> do not cause NaN errors during weighted sum calculation
            for i_e in range(args.e_vocab_size):
                for i_r, i_e2 in KB.e1_view[i_e]:
                    self.r_adjacency[i_e, i_r] += 1
            for i_e in (torch.sum(self.r_adjacency, dim=-1) == 0).nonzero():
                # NO_OP points to self
                self.r_adjacency[i_e, 1] = 1
        self.r_embeddings = nn.Embedding(args.r_vocab_size, args.embedding_size, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.r_embeddings.weight)
        # self.r_embeddings.requires_grad_(False)

        self.policy_size = 3*args.hidden_size if (args.use_entity_embeddings or args.use_neighbourhood_embeddings) \
            else 2*args.hidden_size
        self.rnn_size = 2*args.hidden_size if (args.use_entity_embeddings or args.use_neighbourhood_embeddings) \
            else args.hidden_size

        self.args = args
        self.device = torch.device(args.cuda if self.args.gpu else "cpu")
        self.KB = KB

        # Reasoning machinery

        self.reasoner = StackedRNNCell(args, args.cell_type, self.rnn_size, self.rnn_size, num_layers=args.num_layers)
        self.policy_mlp = nn.Sequential(nn.Linear(self.policy_size, args.hidden_size), nn.ReLU(),
                                        torch.nn.Dropout(args.state_dropout),
                                        nn.Linear(args.hidden_size, self.rnn_size),
                                        torch.nn.Dropout(args.state_dropout))

    def apply_action_dropout_mask(self, action_mask):
        if self.args.action_dropout > 0 and self.training:
            action_dropout_logits = torch.ones_like(action_mask, device=self.device).float()
            action_dropout_logits = self.args.action_dropout * action_dropout_logits
            m = torch.distributions.bernoulli.Bernoulli(logits=action_dropout_logits)
            dropout_mask = m.sample()
            dropout_mask = dropout_mask.byte()
            action_mask = action_mask.__or__(dropout_mask)
            return action_mask
        else:
            return action_mask

    def step(self, args, q, rnn_state, next_rs, next_es, mask, prev_action, epoch=None, step=None):
        rnn_state = self.reasoner(prev_action, rnn_state)

        next_r = self.r_embeddings(next_rs)
        if args.use_entity_embeddings:
            next_e = self.e_embeddings(next_es)
            next_actions = torch.cat([next_r, next_e], dim=-1)
        elif args.use_neighbourhood_embeddings:
            unrolled_e = next_es.view(-1)
            adjacency_piece = self.r_adjacency[unrolled_e, :].to(self.r_embeddings.weight.device)
            rel_weights = self.neighbourhood_embedding_layer(self.r_embeddings.weight).view(1, -1)
            rel_weights = adjacency_piece * torch.exp(rel_weights - rel_weights.max())
            rel_weights = rel_weights / torch.sum(rel_weights, dim=-1, keepdim=True)
            if next_es.dim() == 1:
                next_e = torch.matmul(rel_weights, self.r_embeddings.weight)
            elif next_es.dim() == 2:
                next_e = torch.matmul(rel_weights, self.r_embeddings.weight).view(
                    next_es.shape[0], next_es.shape[1], -1)
            else:
                raise AttributeError("next_e has {} dimensions".format(next_es.dim()))
            next_actions = torch.cat([next_r, next_e], dim=-1)
        else:
            next_actions = next_r

        # chose the h_n
        policy_state = rnn_state[-1][0]


        state_query_concat = torch.cat([policy_state, q], dim=-1)

        output = self.policy_mlp(state_query_concat)
        output = output.unsqueeze(1)
        output = output.repeat(1, args.max_num_actions, 1)
        action_scores = output * next_actions
        action_scores = torch.sum(action_scores, -1)
        # mask = self.apply_action_dropout_mask( mask)
        action_scores = action_scores.masked_fill(mask, -99999.9)

        action_log_probs = torch.log_softmax(action_scores, -1)


        entropy = -torch.softmax(action_scores, -1) * action_log_probs
        entropy = entropy.sum(1).mean()
        actions = Categorical(logits=action_scores)
        assert (torch.sum(actions.probs, dim=1) >= 1 - 1e-5).all()
        chosen_action = actions.sample()
        # if step == 0:
        #     import pdb
        #     pdb.set_trace()

        chosen_e = next_es.gather(-1, chosen_action.view(-1, 1)).view(-1)

        chosen_r = next_rs.gather(-1, chosen_action.view(-1, 1)).view(-1)
        chosen_action_embedding = next_actions[np.arange(next_actions.shape[0]), chosen_action, :]

        loss = torch.nn.functional.cross_entropy(action_scores, chosen_action, reduction="none")
        return rnn_state, chosen_e, chosen_r, chosen_action, action_log_probs, action_scores, chosen_action_embedding, entropy, loss

    def forward(self, start_e, q, e2, epoch=0, end_point=True):

        args = self.args
        KB = self.KB
        log_action_scores = []
        chosen_er = []
        loss = []
        entropy_values = []
        ##RNN init
        full_batch_size = start_e.shape[0]
        true_batch_size = int(full_batch_size / args.beam_size)
        h = torch.tensor(np.zeros(shape=[full_batch_size, self.rnn_size]), dtype=torch.float,device=self.device)
        c = torch.tensor(np.zeros(shape=[full_batch_size, self.rnn_size]), dtype=torch.float,device=self.device)
        rnn_state = (h, c)
        rnn_state = [rnn_state for _ in range(args.num_layers)]

        q_emb = self.r_embeddings(q)
        # concat with previous dummy action
        if args.use_entity_embeddings:
            start_entity_embeddings = self.e_embeddings(start_e)
            prev_action = torch.cat([q_emb, start_entity_embeddings], -1)
        elif args.use_neighbourhood_embeddings:
            unrolled_e = start_e.view(-1)
            adjacency_piece = self.r_adjacency[unrolled_e, :].to(self.r_embeddings.weight.device)
            rel_weights = self.neighbourhood_embedding_layer(self.r_embeddings.weight).view(1, -1)
            rel_weights1 = adjacency_piece * torch.exp(rel_weights - rel_weights.max())
            rel_weights2 = rel_weights1 / torch.sum(rel_weights1, dim=-1, keepdim=True)
            if start_e.dim() == 1:
                start_entity_embeddings = torch.matmul(rel_weights2, self.r_embeddings.weight)
            elif start_e.dim() == 2:
                start_entity_embeddings = torch.matmul(rel_weights2, self.r_embeddings.weight).view(
                    start_e.shape[0], start_e.shape[1], -1)
            else:
                raise AttributeError("start_e has {} dimensions".format(start_e.dim()))
            prev_action = torch.cat([q_emb, start_entity_embeddings], dim=-1)
            assert not torch.isnan(prev_action).any()
        else:
            prev_action = torch.zeros_like(q_emb)
        # import pdb
        # pdb.set_trace()

        current_e = start_e

        if self.training:
            ###Multi step reasoning
            for step_no in range(args.num_steps):
                next_rs, next_es, mask = KB.get_next_e_r_masked(current_e.cpu().numpy().tolist(),
                                                                start_e.cpu().numpy().tolist(),
                                                                e2.cpu().numpy().tolist(),
                                                                q.cpu().numpy().tolist(), step_no)

                next_es = torch.tensor(next_es, dtype=torch.long,device=self.device)
                next_rs = torch.tensor(next_rs, dtype=torch.long,device=self.device)
                mask = torch.tensor(mask, dtype=torch.bool,device=self.device)

                rnn_state, chosen_e, chosen_r, chosen_action, action_log_probs, action_scores, prev_action, entropy, step_loss = self.step(
                    args, q_emb, rnn_state, next_rs, next_es, mask, prev_action, epoch, step = step_no)

                loss.append(step_loss)
                entropy_values.append(entropy)
                log_action_scores.append(action_log_probs.gather(-1, chosen_action.view(-1, 1)).view(1, -1))
                chosen_er.append((chosen_r, chosen_e))
                current_e = chosen_e
            return loss, chosen_er, entropy_values


        else:
            # Beam search
            # import pdb
            # pdb.set_trace()
            # B = Batch
            # k = beam size
            if self.args.beam_search:
                # TODO: add more detailed comments

                all_probs = []

                beam_scores = torch.tensor(np.zeros([true_batch_size * args.beam_size, 1]), dtype=torch.float,device=self.device)
                for step_no in range(args.num_steps):
                    if end_point == True:
                        next_rs, next_es, mask = KB.get_next_e_r_masked(current_e.cpu().numpy().tolist(),
                                                                        start_e.cpu().numpy().tolist(),
                                                                        e2.cpu().numpy().tolist(),
                                                                        q.cpu().numpy().tolist(), step_no)
                    else:
                        next_rs, next_es, mask = KB.get_next_e_r(current_e.cpu().numpy().tolist(),
                                                                        start_e.cpu().numpy().tolist(),
                                                                        q.cpu().numpy().tolist(), step_no)

                    next_es = torch.tensor(next_es, dtype=torch.long,device=self.device) #[B*k, max_num_actions]
                    next_rs = torch.tensor(next_rs, dtype=torch.long,device=self.device) #[B*k, max_num_actions]
                    mask = torch.tensor(mask, dtype=torch.bool,device=self.device)
                    rnn_state, _, _, _, action_log_probs, action_scores, _, _, _ = self.step(args, q_emb, rnn_state,
                                                                                             next_rs, next_es, mask,
                                                                                             prev_action)

                    new_scores = beam_scores + action_log_probs  #  [B*k, max_num_actions] =  [B*k, 1] + [B*k,  max_num_actions]

                    if step_no == 0:
                        top_scores, top_idx = new_scores.sort(dim=-1, descending=True)
                        top_idx = top_idx[:, :args.beam_size]  # [B*k, k]
                        next_elements = torch.tensor(
                            np.array([i * args.beam_size for i in range(true_batch_size)]), dtype=torch.long,device=self.device)

                        next_actions = top_idx[next_elements, :]

                        beam_scores = top_scores[next_elements, :]
                        beam_scores = beam_scores[:, :args.beam_size]
                        beam_scores = beam_scores.contiguous().view(-1, 1)



                    else:

                        new_scores = new_scores.view(true_batch_size, -1) #[B, k*max_num_actions]
                        top_scores, top_idx = new_scores.sort(dim=-1, descending=True) #[B, k*max_num_actions], [B, k*max_num_actions]

                        top_idx = top_idx[:, :args.beam_size] #[B, k]
                        top_scores = top_scores[:, :args.beam_size] #[B, k]
                        beam_scores = top_scores.contiguous().view(-1, 1) #[B*k, 1]


                        x = top_idx // args.max_num_actions #[B, k]
                        next_actions = top_idx % args.max_num_actions #[B, k]
                        next_elements = x.view(-1) + torch.tensor(
                            np.repeat([b * args.beam_size for b in range(true_batch_size)],
                                      args.beam_size), dtype=torch.long,device=self.device) # [B*k,]

                        # Re-arrange memory
                        #TODO: add more detailed comments
                        new_rnn_state = []
                        for layer_no in range(args.num_layers):
                            hc = (rnn_state[layer_no][0][next_elements, :], rnn_state[layer_no][1][next_elements, :])
                            new_rnn_state.append(hc)

                        rnn_state = new_rnn_state

                        next_rs = next_rs[next_elements, :]
                        next_es = next_es[next_elements, :]
                        action_log_probs = action_log_probs[next_elements, :]
                        for j in range(len(chosen_er)):
                            chosen_er[j][0] = chosen_er[j][0][next_elements]
                            chosen_er[j][1] = chosen_er[j][1][next_elements]
                            all_probs[j] = all_probs[j][next_elements]

                    chosen_r = next_rs.gather(-1, next_actions.view(-1, 1)).view(-1) #[B*k]
                    chosen_e = next_es.gather(-1, next_actions.view(-1, 1)).view(-1) #[B*k]
                    chosen_action_prob = action_log_probs.gather(-1, next_actions.view(-1, 1)).view(-1)
                    all_probs.append(chosen_action_prob)

                    r_emb = self.r_embeddings(chosen_r)
                    if args.use_entity_embeddings:
                        e_emb = self.e_embeddings(chosen_e)
                        prev_action = torch.cat([r_emb, e_emb], dim=-1)
                    elif args.use_neighbourhood_embeddings:
                        unrolled_e = chosen_e.view(-1)
                        adjacency_piece = self.r_adjacency[unrolled_e, :].to(self.r_embeddings.weight.device)
                        rel_weights = self.neighbourhood_embedding_layer(self.r_embeddings.weight).view(1, -1)
                        rel_weights = adjacency_piece * torch.exp(rel_weights - rel_weights.max())
                        rel_weights = rel_weights / torch.sum(rel_weights, dim=-1, keepdim=True)
                        if chosen_e.dim() == 1:
                            e_emb = torch.matmul(rel_weights, self.r_embeddings.weight)
                        elif chosen_e.dim() == 2:
                            e_emb = torch.matmul(rel_weights, self.r_embeddings.weight).view(
                                chosen_e.shape[0], chosen_e.shape[1], -1)
                        else:
                            raise AttributeError("chosen_e has {} dimensions".format(chosen_e.dim()))
                        prev_action = torch.cat([r_emb, e_emb], dim=-1)

                    else:
                        prev_action = r_emb

                    chosen_er.append([chosen_r, chosen_e])
                    current_e = chosen_e

                beam_scores = beam_scores.view(true_batch_size, -1)

                # beam_log_probs = torch.softmax(beam_scores, -1).view(-1)
                # return chosen_er, torch.stack(all_probs)
                return chosen_er, torch.stack(all_probs, 1).squeeze(-1)
            else:
                # print("here")

                for step_no in range(args.num_steps):
                    next_rs, next_es, mask = KB.get_next_e_r_masked(current_e.cpu().numpy().tolist(),
                                                                    start_e.cpu().numpy().tolist(),
                                                                    e2.cpu().numpy().tolist(),
                                                                    q.cpu().numpy().tolist(), step_no)
                    next_es = torch.tensor(next_es, dtype=torch.long,device=self.device)
                    next_rs = torch.tensor(next_rs, dtype=torch.long,device=self.device)
                    mask = torch.tensor(mask, dtype=torch.bool,device=self.device)

                    rnn_state, chosen_e, chosen_r, chosen_action, action_log_probs, action_scores, prev_action, entropy, step_loss = self.step(
                        args, q_emb,
                        rnn_state,
                        next_rs,
                        next_es,
                        mask,
                        prev_action,
                        epoch)

                    log_action_scores.append(action_scores.gather(-1, chosen_action.view(-1, 1)).view(1, -1))
                    chosen_er.append([chosen_r, chosen_e])
                    current_e = chosen_e
                log_probs = torch.stack(log_action_scores, -1).sum(-1).view(true_batch_size, -1)
                per_path_scores = torch.stack(log_action_scores, -1).view(-1, args.num_steps)
                sorted_log_probs, idx = log_probs.sort(-1, descending=True)

                per_path_scores = per_path_scores[idx, :]

                idx = idx.view(-1) + torch.tensor(
                    np.repeat([b * args.beam_size for b in range(true_batch_size)], args.beam_size), dtype=torch.long,device=self.device)

                for j in range(len(chosen_er)):
                    chosen_er[j][0] = chosen_er[j][0][idx]
                    chosen_er[j][1] = chosen_er[j][1][idx]
                log_probs = sorted_log_probs.view(-1, 1)
                return chosen_er, per_path_scores

