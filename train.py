import numpy as np
import torch
import logging
from utils import AverageMeter, Timer
import os
from options import read_options
import sys
import json
from Model import KBCompleter
from collections import defaultdict
from utils import Printer
logger = logging.getLogger()
from kb import KB, KB_dataset
from KGE import ConvE_wn18rr_model, ConvE_fb15k_model, Conve_base_model



def filter(e1,r,e2):
    # males = set([e.strip() for e in open("/home/shdhulia/KB/datasets/FB/male").readlines()])
    # females = set([e.strip() for e in open("/home/shdhulia/KB/datasets/FB/female").readlines()])
    if r == "/people/person/profession":
        return True
    else:
        return False

def mfilter(e1,r,e2):
    males = [e.strip() for e in open("/home/shdhulia/Projects/KB/datasets/FB/male").readlines()]
    # if r == "/people/person/profession" and e1 in males:
    if e1 in males:
        return True
    else:
        return False
def ffilter(e1,r,e2):
    females = [e.strip() for e in open("/home/shdhulia/Projects/KB/datasets/FB/female").readlines()]
    # if r == "/people/person/profession" and e1 in females:
    if e1 in females:
        return True
    else:
        return False


class Trainer(object):
    def __init__(self, args):
        '''
        collects the data and creates the samplers
        :param args:
        '''
        self.args = args
        self.device = torch.device(args.cuda if self.args.gpu else "cpu")
        self.stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0}

        self.KB = KB(args)
        args.e_vocab_size = len(self.KB.e_vocab)
        args.r_vocab_size = len(self.KB.r_vocab)
        if args.load_model != "":
            if args.run_on_new_args:
                self.model = KBCompleter.load(args.load_model, new_args=args, new_KB=self.KB, use_new_args=True)
            else:
                self.model = KBCompleter.load(args.load_model, new_args=args, use_new_args=False)
        else:
            self.model = KBCompleter(args, self.KB)
        self.train_data = KB_dataset(args, args.train, self.KB.e_vocab, self.KB.r_vocab, filter=False, filtering_function=filter)
        self.dev_data = KB_dataset(args, args.dev, self.KB.e_vocab, self.KB.r_vocab, train=False, filter=False, filtering_function=mfilter)
        self.test_data = KB_dataset(args, args.test, self.KB.e_vocab, self.KB.r_vocab, train=False, filter=False, filtering_function=mfilter)
        self.model.init_optimizer()
        self.best_metric = 0.0


        # Use the GPU?
        if args.gpu:
            self.model.cuda()

        train_sampler = torch.utils.data.sampler.RandomSampler(self.train_data)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(self.dev_data)
        test_sampler = torch.utils.data.sampler.SequentialSampler(self.test_data)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=args.data_workers,
            collate_fn=self.train_data.batchify,
            pin_memory=args.cuda,
        )
        self.dev_loader = torch.utils.data.DataLoader(
            self.dev_data,
            sampler=dev_sampler,
            batch_size=args.eval_batch_size,
            num_workers=args.data_workers,
            collate_fn=self.dev_data.batchify,
            pin_memory=args.cuda,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            sampler=test_sampler,
            batch_size=args.eval_batch_size,
            num_workers=args.data_workers,
            collate_fn=self.test_data.batchify,
            pin_memory=args.cuda,
        )

    def train(self):
        train_loss = AverageMeter()
        avg_reward_meter = AverageMeter()
        epoch_time = Timer()
        self.stats['epoch'] += 1
        data_loader = self.train_loader
        for idx, ex in enumerate(data_loader):
            # Run one epoch
            avg_reward, loss, batch_size, reward = self.model.update(ex, self.stats['epoch'])
            train_loss.update(loss, batch_size)
            avg_reward_meter.update(avg_reward, batch_size)

            if idx % args.display_iter == 0:
                logger.info('train: Epoch = %d | iter = %d/%d | ' %
                            (self.stats['epoch'], idx, len(data_loader)) +
                            'loss = %.2f | avg_reward = %.4f | no_hits = %d | elapsed time = %.2f (s)' %
                            (train_loss.avg, avg_reward, sum(reward), self.stats['timer'].time()))
                train_loss.reset()

            # logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
            #             (self.stats['epoch'], epoch_time.time()))




    def eval(self, data_loader, is_test=False):
        epoch_time = Timer()
        hits_at_1 = AverageMeter()
        hits_at_3 = AverageMeter()
        hits_at_10 = AverageMeter()
        auc = AverageMeter()
        if is_test and args.print_paths:
            printer = Printer(args)
        for idx, ex in enumerate(data_loader):
            # Run one epoch

            predicted_e2, e2, _ = self.model.predict(ex, self.stats['epoch'], is_test, printer=printer if is_test else None)

            batch_size = e2.shape[0]

            for b in range(batch_size):
                correct = e2[b, 0]
                already_seen = set([])
                position = 0
                found = False
                # import pdb
                # pdb.set_trace()
                for r in range(args.beam_size):
                    predicted = predicted_e2[b, r]
                    if predicted != correct:
                        if predicted not in already_seen:
                            already_seen.add(predicted)
                            position += 1
                        else:
                            continue
                    else:
                        found = True
                        break
                if position < 1 and found:
                    hits_at_1.update(1.0)
                else:
                    hits_at_1.update(0.0)
                if position < 3 and found:
                    hits_at_3.update(1.0)
                else:
                    hits_at_3.update(0.0)
                if position < 10 and found:
                    hits_at_10.update(1.0)
                else:
                    hits_at_10.update(0.0)
                if found:
                    auc.update(1/(position+1))
                else:
                    auc.update(0)
                # if position
        if args.print_paths and is_test:
            printer.print()

        logger.info("Hits @ 1: {}".format(hits_at_1.avg))
        logger.info("Hits @ 3: {}".format(hits_at_3.avg))
        logger.info("Hits @10: {}".format(hits_at_10.avg))
        logger.info("AUC     : {}".format(auc.avg))
        if hits_at_10.avg >= self.best_metric and is_test == False:
            self.best_metric = hits_at_10.avg
            self.model.save(args.output_dir+"/model")

    def per_relation_eval(self, data_loader, is_test=False):
        assert is_test == True
        epoch_time = Timer()
        hits_at_1 = defaultdict(float)
        hits_at_3 = defaultdict(float)
        hits_at_10 = defaultdict(float)
        auc = defaultdict(float)
        counts = defaultdict(int)
        if is_test:
            printer = Printer(args)
        for idx, ex in enumerate(data_loader):
            # Run one epoch
            _, r_batch, _ = ex
            # import pdb
            # pdb.set_trace()

            predicted_e2, e2, _ = self.model.predict(ex, self.stats['epoch'], is_test, printer=printer if is_test else None)

            batch_size = e2.shape[0]

            for b in range(batch_size):
                correct = e2[b, 0]
                rel = r_batch.numpy().tolist()[b]
                counts[rel] += 1
                already_seen = set([])
                position = 0
                found = False
                # import pdb
                # pdb.set_trace()
                for r in range(args.beam_size):
                    predicted = predicted_e2[b, r]
                    if predicted != correct:
                        if predicted not in already_seen:
                            already_seen.add(predicted)
                            position += 1
                        else:
                            continue
                    else:
                        found = True
                        break
                if position < 1 and found:
                    hits_at_1[rel] += 1.0

                if position < 3 and found:
                    hits_at_3[rel] += 1.0

                if position < 10 and found:
                    hits_at_10[rel] += 1.0

                if found:
                    auc[rel] += (1 / (position + 1))

                # if position
        for rel, count in counts.items():
            hits_at_1[rel] /= counts[rel]
            hits_at_3[rel] /= counts[rel]
            hits_at_10[rel] /= counts[rel]
            auc[rel] /= counts[rel]

            rel_name = self.KB.r_vocab[rel]
            logger.info("for relation {}".format(rel_name))
            logger.info("Hits @ 1: {}".format(hits_at_1[rel]))
            logger.info("Hits @ 3: {}".format(hits_at_3[rel]))
            logger.info("Hits @10: {}".format(hits_at_10[rel]))
            logger.info("AUC     : {}".format(auc[rel]))
        if args.print_paths and is_test:
            printer.print()




            # if idx % args.display_iter == 0:
            #     logger.info('train: Epoch = %d | iter = %d/%d | ' %
            #                 (global_stats['epoch'], idx, len(data_loader)) +
            #                 'loss = %.2f | elapsed time = %.2f (s)' %
            #                 (train_loss.avg, global_stats['timer'].time()))
            #     train_loss.reset()
            #
            # logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
            #             (global_stats['epoch'], epoch_time.time()))


    def run_path_find_ad_hoc(self, e1, r, e2=None):

        '''

        :param e1: input e1
        :param r: input query relation
        :param e2: input correct e2. If not provided no masking takes place
        :return: chosen_er, decoded_chosen_er : [path_len, 2:(r, e)] note the e,rs can be decoded to the string using self.KB.e_vocab and self.KB.r_vocab
        e1 -> (r,e)_1 -> (r,e)_2 -> (r,e)_3
        Function outputs the poilicy paths for given e1, r.

        example:
        args = json.load("model_to_load/args.json")
        args.beam_size = 40 #optional to change beam size
        trainer = Trainer(args)

        chosen_er = trainer.run_path_find_ad_hoc("/m/05r6t", "/music/genre/artists")



        '''



        e1 = [self.KB.e_vocab[e1], ]* self.args.beam_size
        r = [self.KB.r_vocab[r], ]* self.args.beam_size
        e1 = torch.tensor(e1, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.long)


        if e2 != None:
            e2 = [self.KB.e_vocab[e2], ] * self.args.beam_size
            e2 = torch.tensor(e2, dtype=torch.long)

        if self.args.gpu:
            e1, r = e1.to(self.device), r.to(self.device)
            if e2:
                e2 = e2.to(self.device)

        chosen_er, _ = self.model.get_paths(e1,r,e2, self.stats['epoch'] )
        chosen_er = [[t[0].detach().cpu().numpy().tolist(), t[1].detach().cpu().numpy().tolist()] for t in chosen_er]
        decoded_chosen_er = [[[self.KB.r_vocab[r] for r in ER[0]],  [self.KB.e_vocab[e] for e in ER[1]]] for ER in chosen_er]
        return chosen_er, decoded_chosen_er




def main(args):
    # TRAIN/VALID LOOP
    trainer = Trainer(args)

    if args.trainkge:
        emb_dir = os.path.join("datasets/", args.dataset, "embeddings/")
        if args.dataset == "WN18RR":
            kge_model = ConvE_wn18rr_model(args, trainer.KB)
        if args.dataset == "FB15K-237":
            kge_model = ConvE_fb15k_model(args, trainer.KB)
        if args.dataset == "nell-995":
            kge_model = Conve_base_model(args, trainer.KB)
        kge_model.model.to(args.cuda)
        mrr, hit1, hit3, hit10 = kge_model.train()
        print("MRR : {0}, Hit@1 : {1}, Hit@3 : {2}, Hit@10 : {3}".format(mrr, hit1, hit3, hit10))
        e_emb = kge_model.model.entity_representations[0]()
        r_emb = kge_model.model.relation_representations[0]()
        torch.save(e_emb, emb_dir + "node_embedding.pt")
        torch.save(r_emb, emb_dir + "rel_embedding.pt")
        print("Nodes embeddings : {}, Rel embeddings : {}".format(e_emb.size(), r_emb.size()))
        print("Embeddings saved at {}".format(emb_dir))
        return

    # trainer.run_path_find_ad_hoc("/m/05r6t", "/music/genre/artists")

    if args.load_model != "":
        if args.per_relation_scores == 0:

            trainer.eval(trainer.test_loader, is_test=True)
        else:
            trainer.per_relation_eval(trainer.test_loader, is_test=True)
    else:
        for i in range(args.num_epochs):
            trainer.train()
            if i%args.eval_every == 0:
                trainer.eval(trainer.dev_loader)
        trainer.model = trainer.model.load(args.output_dir+"/model")
        if args.gpu:
            trainer.model.cuda()
        if args.per_relation_scores == 0:
            trainer.eval(trainer.test_loader, is_test=True)
        else:
            trainer.per_relation_eval(trainer.test_loader, is_test=True)




    # if args.eval_only:
    #     logger.info("Eval only mode")
    #     result = validate_official(args, dev_loader, model, stats, None, None, None,
    #                                ground_truths_map=dev_ground_truths_map, official_eval_output=True)
    #     logger.info("Exiting...")
    #     sys.exit(0)

    # logger.info('Starting training...')
    # for epoch in range(0, args.num_epochs):
    #     # stats['epoch'] = epoch
    #     # Train
    #     trainer.train()
    #     # Validate unofficial (train)
    #     # validate_unofficial(args, train_loader, model, stats, mode='train')
    #     # Validate unofficial (dev)
    #     # result = validate_unofficial(args, dev_loader, model, stats, mode='dev')
    #     # Validate official
    #     if args.official_eval:
    #         result = validate_official(args, dev_loader, model, stats, None, None, None, ground_truths_map=dev_ground_truths_map)
    #         # result = validate_official(args, train_loader, model, stats, None, None, None, ground_truths_map=train_ground_truths_map)
    #
    #     # Save best valid
    #     if result[args.valid_metric] > stats['best_valid']:
    #         logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
    #                     (args.valid_metric, result[args.valid_metric],
    #                      stats['epoch'], model.updates))
    #         model.save(args.model_file)
    #         stats['best_valid'] = result[args.valid_metric]


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    args = read_options()


    # Set cuda
    if args.gpu:
        torch.device(args.cuda)
        print(torch.cuda.device_count())
        print("Device used {}".format(torch.cuda.current_device()))
    else:
        torch.device("cpu")

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    args.log_file = os.path.join(args.output_dir, "logs.txt")
    if args.log_file:

        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)







