import csv
from pykeen.triples import CoreTriplesFactory
import torch
from torch.optim import Adam, Adagrad
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import ConvE
from pykeen.training import LCWATrainingLoop
from pykeen.losses import BCEAfterSigmoidLoss
from kb import KB


class ConvE_wn18rr_model():
    def __init__(self, args, kb: KB):
        self.kb = kb
        self.ent_vocab = self.kb.e_vocab
        self.rel_vocab = self.kb.r_vocab
        self.emb_size = args.embedding_size
        self.mapped_triples = torch.tensor(self.kb.kb)
        self.triples_factory = CoreTriplesFactory(self.mapped_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=False)
        self.train_triples, self.val_triples, self.test_triples = self.triples_factory.split(ratios=[0.8, 0.1, 0.1])
        self.model = ConvE(triples_factory=self.train_triples, embedding_dim=self.emb_size, loss="CrossEntropyLoss", feature_map_dropout=0.21969167540833145,
                           input_dropout=0.3738110367324488, input_channels=None, kernel_height=3, kernel_width=3,
                           output_channels=27, output_dropout=0.4598078311847786, )
        self.optimizer = Adam(self.model.get_grad_params(), lr=0.0015640153246253687, weight_decay=0.0)
        self.training_loop = LCWATrainingLoop(model=self.model, triples_factory=self.train_triples,
                                              optimizer=self.optimizer)
        self.evaluator = RankBasedEvaluator()

    def train(self):
        losses = self.training_loop.train(triples_factory=self.train_triples, num_epochs=601, batch_size=256, label_smoothing=0.003261077338126352, 
                                          use_tqdm_batch=False)
        val_metrics = self.evaluator.evaluate(model = self.model, mapped_triples=self.test_triples.mapped_triples,
                                              additional_filter_triples=[self.train_triples.mapped_triples,self.val_triples.mapped_triples])
        val_metrics = val_metrics.to_flat_dict()
        mrr = val_metrics["both.realistic.inverse_harmonic_mean_rank"]
        hit1 = val_metrics["both.realistic.hits_at_1"]
        hit3 = val_metrics["both.realistic.hits_at_3"]
        hit10 = val_metrics["both.realistic.hits_at_10"]
        return mrr, hit1, hit3, hit10

class ConvE_fb15k_model():
    def __init__(self, args, kb: KB):
        self.kb = kb
        self.ent_vocab = self.kb.e_vocab
        self.rel_vocab = self.kb.r_vocab
        self.emb_size = args.embedding_size
        self.mapped_triples = torch.tensor(self.kb.kb)
        self.triples_factory = CoreTriplesFactory(self.mapped_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=False)
        self.train_triples, self.val_triples, self.test_triples = self.triples_factory.split(ratios=[0.8, 0.1, 0.1])
        self.model = ConvE(triples_factory=self.train_triples, embedding_dim=self.emb_size, loss="CrossEntropyLoss", feature_map_dropout=0.38074998430562207,
                           input_dropout=0.481083618149555, input_channels=None, kernel_height=3, kernel_width=3,
                           output_channels=56, output_dropout=0.4920249242322924, )
        self.optimizer = Adam(self.model.get_grad_params(), lr=0.0052417396207321025, weight_decay=0.0)
        self.training_loop = LCWATrainingLoop(model=self.model, triples_factory=self.train_triples,
                                              optimizer=self.optimizer)
        self.evaluator = RankBasedEvaluator()

    def train(self):
        losses = self.training_loop.train(triples_factory=self.train_triples, num_epochs=401, batch_size=256, label_smoothing=0.05422578918650805, 
                                          use_tqdm_batch=False)
        val_metrics = self.evaluator.evaluate(model = self.model, mapped_triples=self.test_triples.mapped_triples,
                                              additional_filter_triples=[self.train_triples.mapped_triples,self.val_triples.mapped_triples])
        val_metrics = val_metrics.to_flat_dict()
        mrr = val_metrics["both.realistic.inverse_harmonic_mean_rank"]
        hit1 = val_metrics["both.realistic.hits_at_1"]
        hit3 = val_metrics["both.realistic.hits_at_3"]
        hit10 = val_metrics["both.realistic.hits_at_10"]
        return mrr, hit1, hit3, hit10
    
class Conve_base_model():
    def __init__(self, args, kb: KB):
        self.kb = kb
        self.ent_vocab = self.kb.e_vocab
        self.rel_vocab = self.kb.r_vocab
        self.emb_size = args.embedding_size
        self.mapped_triples = torch.tensor(self.kb.kb)
        self.triples_factory = CoreTriplesFactory(self.mapped_triples, len(self.ent_vocab), len(self.rel_vocab),
                                                  self.ent_vocab, self.rel_vocab, create_inverse_triples=False)
        self.train_triples, self.val_triples, self.test_triples = self.triples_factory.split(ratios=[0.8, 0.1, 0.1])
        self.model = ConvE(triples_factory=self.train_triples, embedding_dim=self.emb_size)
        self.optimizer = Adam(self.model.get_grad_params())
        self.training_loop = LCWATrainingLoop(model=self.model, triples_factory=self.train_triples,
                                              optimizer=self.optimizer)
        self.evaluator = RankBasedEvaluator()

    def train(self):
        losses = self.training_loop.train(triples_factory=self.train_triples, num_epochs=1000, batch_size=256, 
                                          use_tqdm_batch=False)
        val_metrics = self.evaluator.evaluate(model = self.model, mapped_triples=self.test_triples.mapped_triples,
                                              additional_filter_triples=[self.train_triples.mapped_triples,self.val_triples.mapped_triples])
        val_metrics = val_metrics.to_flat_dict()
        mrr = val_metrics["both.realistic.inverse_harmonic_mean_rank"]
        hit1 = val_metrics["both.realistic.hits_at_1"]
        hit3 = val_metrics["both.realistic.hits_at_3"]
        hit10 = val_metrics["both.realistic.hits_at_10"]
        return mrr, hit1, hit3, hit10