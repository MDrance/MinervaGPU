import csv
from pykeen.triples import CoreTriplesFactory
import torch
from torch.optim import Adam, Adagrad
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import ConvE
from pykeen.training import LCWATrainingLoop
from pykeen.losses import BCEAfterSigmoidLoss
from kb import KB, KB_dataset


class ConvE_wn18rr_model():
    def __init__(self, args, kb: KB, train_set: KB_dataset, test_set: KB_dataset):
        self.args = args
        self.kb = kb
        self.ent_vocab = self.kb.e_vocab
        self.rel_vocab = self.create_rvocab()
        self.emb_size = args.embedding_size
        self.train_triples = torch.tensor(train_set.kb)
        self.test_triples = torch.tensor(test_set.kb)
        self.train_triples_factory = CoreTriplesFactory(self.train_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=True)
        self.test_triples_factory = CoreTriplesFactory(self.test_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=False)

        self.model = ConvE(triples_factory=self.train_triples_factory, embedding_dim=self.emb_size, loss="CrossEntropyLoss", feature_map_dropout=0.21969167540833145,
                           input_dropout=0.3738110367324488, input_channels=None, kernel_height=3, kernel_width=3,
                           output_channels=27, output_dropout=0.4598078311847786, )
        self.optimizer = Adam(self.model.get_grad_params(), lr=0.0015640153246253687, weight_decay=0.0)
        self.training_loop = LCWATrainingLoop(model=self.model, triples_factory=self.train_triples_factory,
                                              optimizer=self.optimizer)
        self.evaluator = RankBasedEvaluator()

    def train(self):
        losses = self.training_loop.train(triples_factory=self.train_triples_factory, num_epochs=601, batch_size=256, label_smoothing=0.003261077338126352, 
                                          use_tqdm_batch=False)
        val_metrics = self.evaluator.evaluate(model = self.model, mapped_triples=self.test_triples_factory.mapped_triples)
        val_metrics = val_metrics.to_flat_dict()
        mrr = val_metrics["both.realistic.inverse_harmonic_mean_rank"]
        hit1 = val_metrics["both.realistic.hits_at_1"]
        hit3 = val_metrics["both.realistic.hits_at_3"]
        hit10 = val_metrics["both.realistic.hits_at_10"]
        return mrr, hit1, hit3, hit10
    
    def create_rvocab(self) -> dict:
        all_keys = [k for k in self.kb.r_vocab.tok2ind.keys()][2:]
        all_values = [k for k in self.kb.r_vocab.tok2ind.values()][2:]
        half = len(all_keys)//2
        frw_keys = all_keys[:half]
        frw_values = all_values[:half]
        return {k : v for k, v in zip(frw_keys, frw_values)}
    
    def organize_rvocab(self):
        special_token_emb = torch.nn.Embedding(2, self.args.embedding_size)
        rel_emb = self.model.relation_representations[0]().data
        return torch.cat((special_token_emb.weight.data, rel_emb[0::2], rel_emb[1::2]), dim=0)
        

class ConvE_fb15k_model():
    def __init__(self, args, kb: KB, train_set: KB_dataset, test_set: KB_dataset):
        self.args = args
        self.kb = kb
        self.ent_vocab = self.kb.e_vocab
        self.rel_vocab = self.rel_vocab = self.create_rvocab()
        self.emb_size = args.embedding_size
        self.train_triples = torch.tensor(train_set.kb)
        self.test_triples = torch.tensor(test_set.kb)
        self.train_triples_factory = CoreTriplesFactory(self.train_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=True)
        self.test_triples_factory = CoreTriplesFactory(self.test_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=False)
        
        self.model = ConvE(triples_factory=self.train_triples_factory, embedding_dim=self.emb_size, loss="CrossEntropyLoss", feature_map_dropout=0.38074998430562207,
                           input_dropout=0.481083618149555, input_channels=None, kernel_height=3, kernel_width=3,
                           output_channels=56, output_dropout=0.4920249242322924, )
        self.optimizer = Adam(self.model.get_grad_params(), lr=0.0052417396207321025, weight_decay=0.0)
        self.training_loop = LCWATrainingLoop(model=self.model, triples_factory=self.train_triples_factory,
                                              optimizer=self.optimizer)
        self.evaluator = RankBasedEvaluator()

    def train(self):
        losses = self.training_loop.train(triples_factory=self.train_triples_factory, num_epochs=401, batch_size=256, label_smoothing=0.05422578918650805, 
                                          use_tqdm_batch=False)
        val_metrics = self.evaluator.evaluate(model = self.model, mapped_triples=self.test_triples_factory.mapped_triples)
        val_metrics = val_metrics.to_flat_dict()
        mrr = val_metrics["both.realistic.inverse_harmonic_mean_rank"]
        hit1 = val_metrics["both.realistic.hits_at_1"]
        hit3 = val_metrics["both.realistic.hits_at_3"]
        hit10 = val_metrics["both.realistic.hits_at_10"]
        return mrr, hit1, hit3, hit10
    
    def create_rvocab(self) -> dict:
        all_keys = [k for k in self.kb.r_vocab.tok2ind.keys()][2:]
        all_values = [k for k in self.kb.r_vocab.tok2ind.values()][2:]
        half = len(all_keys)//2
        frw_keys = all_keys[:half]
        frw_values = all_values[:half]
        return {k : v for k, v in zip(frw_keys, frw_values)}
    
    def organize_rvocab(self):
        special_token_emb = torch.nn.Embedding(2, self.args.embedding_size)
        rel_emb = self.model.relation_representations[0]().data
        return torch.cat((special_token_emb.weight.data, rel_emb[0::2], rel_emb[1::2]), dim=0)
    
class Conve_base_model():
    def __init__(self, args, kb: KB, train_set: KB_dataset, test_set: KB_dataset):
        self.args = args
        self.kb = kb
        self.ent_vocab = self.kb.e_vocab
        self.rel_vocab = self.rel_vocab = self.create_rvocab()
        self.emb_size = args.embedding_size
        self.train_triples = torch.tensor(train_set.kb)
        self.test_triples = torch.tensor(test_set.kb)
        self.train_triples_factory = CoreTriplesFactory(self.train_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=True)
        self.test_triples_factory = CoreTriplesFactory(self.test_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=False)
        
        self.model = ConvE(triples_factory=self.train_triples_factory, embedding_dim=self.emb_size)
        self.optimizer = Adam(self.model.get_grad_params())
        self.training_loop = LCWATrainingLoop(model=self.model, triples_factory=self.train_triples_factory,
                                              optimizer=self.optimizer)
        self.evaluator = RankBasedEvaluator()

    def train(self):
        losses = self.training_loop.train(triples_factory=self.train_triples_factory, num_epochs=500, batch_size=256, 
                                          use_tqdm_batch=False)
        val_metrics = self.evaluator.evaluate(model = self.model, mapped_triples=self.test_triples_factory.mapped_triples)
        val_metrics = val_metrics.to_flat_dict()
        mrr = val_metrics["both.realistic.inverse_harmonic_mean_rank"]
        hit1 = val_metrics["both.realistic.hits_at_1"]
        hit3 = val_metrics["both.realistic.hits_at_3"]
        hit10 = val_metrics["both.realistic.hits_at_10"]
        return mrr, hit1, hit3, hit10
    
    def create_rvocab(self) -> dict:
        all_keys = [k for k in self.kb.r_vocab.tok2ind.keys()][2:]
        all_values = [k for k in self.kb.r_vocab.tok2ind.values()][2:]
        half = len(all_keys)//2
        frw_keys = all_keys[:half]
        frw_values = all_values[:half]
        return {k : v for k, v in zip(frw_keys, frw_values)}
    
    def organize_rvocab(self):
        special_token_emb = torch.nn.Embedding(2, self.args.embedding_size)
        rel_emb = self.model.relation_representations[0]().data
        return torch.cat((special_token_emb.weight.data, rel_emb[0::2], rel_emb[1::2]), dim=0)
    
