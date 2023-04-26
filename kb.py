from collections import  defaultdict
import numpy as np
import unicodedata
import torch
from torch.utils.data import Dataset, DataLoader


class Dictionary(object):
    NULL = '<NULL>'
    NO_OP = 'NO_OP'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.NO_OP: 1}
        self.ind2tok = {0: self.NULL, 1: self.NO_OP}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.NULL)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.NULL))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', 'NO_OP'}]
        return tokens

class KB():
    def __init__(self, args):

        self.e1_view = defaultdict(list)
        self.er_view = defaultdict(list)
        self.kb = []
        self.r_view = defaultdict(list)
        self.args = args
        self.page_rank = {}
        self.e_vocab = Dictionary()
        self.r_vocab = Dictionary()
        self.max_num_actions = args.max_num_actions
        self.sort_by_page_rank = False
        self.dummy_action = self.r_vocab["dummy_action"]
        with open(args.graph) as tsv:
            for line in tsv:
                try:
                    e1,r,e2 = line.strip().split("\t")
                except:
                    import pdb
                    pdb.set_trace()
                self.e_vocab.add(e1)
                self.e_vocab.add(e2)
                self.r_vocab.add(r)

                e1 = self.e_vocab[e1]
                e2 = self.e_vocab[e2]
                r = self.r_vocab[r]
                self.r_view[int(r)].append((int(e1), int(e2)))
                self.e1_view[int(e1)].append((int(r), int(e2)))
                self.er_view[(int(e1), int(r))].append(int(e2)) #used for masking out
                self.kb.append((e1,r,e2))
        # adding no op option to all
        for e1 in self.e_vocab:
            if e1 in [0,1]:
                continue
            e1 = self.e_vocab[e1]
            self.e1_view[e1].insert(0, (1, e1))
            self.er_view[(e1, 1)].insert(0, e1)


        with open(args.dev) as tsv:
            for line in tsv:
                e1, r, e2 = line.strip().split("\t")
                self.e_vocab.add(e1)
                self.e_vocab.add(e2)
                self.r_vocab.add(r)
                e1 = self.e_vocab[e1]
                e2 = self.e_vocab[e2]
                r = self.r_vocab[r]
                self.er_view[(int(e1), int(r))].append(int(e2))  # used for masking out

        if args.dataset == "nell-995":
            with open(args.test) as tsv:
                for line in tsv:
                    e1, r, e2 = line.strip().split("\t")
                    self.e_vocab.add(e1)
                    self.e_vocab.add(e2)
                    self.r_vocab.add(r)
                    e1 = self.e_vocab[e1]
                    e2 = self.e_vocab[e2]
                    r = self.r_vocab[r]
                    self.er_view[(int(e1), int(r))].append(int(e2))  # used for masking out
        if args.page_rank != None:
            with open(args.page_rank) as pgrnk:
                self.sort_by_page_rank = True
                self.page_rank[0] = 0.0
                for line in pgrnk:
                    e, score = line.strip().split()
                    self.page_rank[self.e_vocab[e]] = float(score[1:])






    def get_next_e_r(self,  curr_es, e1_b, qr_b, step_no):

        next_e = np.zeros([len(curr_es), self.max_num_actions], dtype=int)
        next_r = np.zeros([len(curr_es), self.max_num_actions], dtype=int)
        mask = np.ones([len(curr_es), self.max_num_actions], dtype=bool)

        for batch_count, curr_e in enumerate(curr_es):
            qr = qr_b[batch_count]
            e1 = e1_b[batch_count]
            outgoing_edges = self.e1_view[curr_e]
            # if self.e_vocab['eritrea'] == e1 and step_no == 1:
            #     import pdb
            #     pdb.set_trace()

            for action_count, tup in enumerate(outgoing_edges):
                if action_count >= self.max_num_actions:
                    break
                r, e = tup
                if curr_e == e1 and qr == r:
                    continue

                next_e[batch_count, action_count] = e
                next_r[batch_count, action_count] = r
                mask[batch_count, action_count] = 0
                if self.sort_by_page_rank:
                    sorted_idx = np.argsort([self.page_rank[entity] for entity in next_e[batch_count, :]])
                    next_e[batch_count,] = next_e[batch_count, sorted_idx]
                    next_r[batch_count,] = next_r[batch_count, sorted_idx]
                    mask[batch_count,] = mask[batch_count, sorted_idx]

            # if step_no == 1 and curr_e == 4:
            #     import pdb
            #     pdb.set_trace()
            # next_e[batch_count, :] = next_e[batch_count, random_shufffle]
            # next_r[batch_count, :] = next_r[batch_count, random_shufffle]
            # mask[batch_count, :] = mask[batch_count, random_shufffle]
            # if 50 == e1:
            #     import pdb
            #     pdb.set_trace()
            # import pdb
            # pdb.set_trace()
        return next_r, next_e, mask

    def get_next_e_r_masked(self, curr_es, e1_b, e2_b, qr_b, step_no):

        next_e = np.zeros([len(curr_es), self.max_num_actions], dtype=int)
        next_r = np.zeros([len(curr_es), self.max_num_actions], dtype=int)
        mask = np.ones([len(curr_es), self.max_num_actions], dtype=bool)

        for batch_count, curr_e in enumerate(curr_es):
            e2 = e2_b[batch_count]
            qr = qr_b[batch_count]
            e1 = e1_b[batch_count]
            outgoing_edges = self.e1_view[curr_e]
            # if self.e_vocab['eritrea'] == e1 and step_no == 1:
            #     import pdb
            #     pdb.set_trace()


            for action_count, tup in enumerate(outgoing_edges):
                if action_count >= self.max_num_actions:
                    break
                r , e = tup
                if step_no == self.args.num_steps -1 and e in self.er_view[(e1,qr)] and e != e2:
                    continue
                if curr_e == e1 and qr == r and e2 == e:
                    continue
                next_e[batch_count, action_count] = e
                next_r[batch_count, action_count] = r
                mask[batch_count, action_count] = 0
                if self.sort_by_page_rank:
                    sorted_idx = np.argsort([self.page_rank[entity] for entity in next_e[batch_count, :]])
                    next_e[batch_count, ] = next_e[batch_count, sorted_idx]
                    next_r[batch_count, ] = next_r[batch_count, sorted_idx]
                    mask[batch_count, ] = mask[batch_count, sorted_idx]

            # if step_no == 1 and curr_e == 4:
            #     import pdb
            #     pdb.set_trace()
            # next_e[batch_count, :] = next_e[batch_count, random_shufffle]
            # next_r[batch_count, :] = next_r[batch_count, random_shufffle]
            # mask[batch_count, :] = mask[batch_count, random_shufffle]
            # if 50 == e1:
            #     import pdb
            #     pdb.set_trace()
            # import pdb
            # pdb.set_trace()
        return next_r, next_e, mask

    def follow_path(self, start_entities, relations):
        current_entities = start_entities
        for step_no, next_relations in enumerate(relations):
            next_entities = []
            next_relations = next_relations.cpu().numpy().tolist()
            for e, r in zip(current_entities, next_relations):
                next_es = self.er_view[(e,r)]
                next_es.append(0)
                next_e = np.random.choice(len(next_es), 1)
                next_e = next_es[next_e[0]]
                next_entities.append(next_e)
            current_entities = next_entities
        return current_entities



class KB_dataset(Dataset):
    def __init__(self, args, file, e_vocab, r_vocab, train=True, filter = False, filtering_function = None):
        self.args = args
        self.kb = []
        self.e_vocab = e_vocab
        self.r_vocab = r_vocab
        self.max_num_actions = args.max_num_actions
        self.rollouts = self.args.num_rollouts if train else self.args.beam_size
        with open(file) as tsv:
            for line in tsv:
                e1,r,e2 = line.strip().split("\t")
                if filter:
                    if not filtering_function(e1,r,e2):
                        continue

                e1 = self.e_vocab[e1]
                e2 = self.e_vocab[e2]
                r = self.r_vocab[r]

                self.kb.append((e1,r,e2))


    def __len__(self):
        return len(self.kb)

    def __getitem__(self, idx):
        e1,r,e2 = self.kb[idx]
        return e1,r,e2

    def batchify(self, batch):
        e1 = [e for ex in batch for e in [ex[0],] * self.rollouts]
        r = [e for ex in batch for e in [ex[1],] * self.rollouts]
        e2 = [e for ex in batch for e in [ex[2],] * self.rollouts]
        e1 = torch.tensor(e1, dtype=int)
        r = torch.tensor(r, dtype=int)
        e2 = torch.tensor(e2, dtype=int)
        return e1, r, e2
