import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy

from zmq import device
from MINERVA import  Minerva
from collections import defaultdict
from utils import Printer



logger = logging.getLogger(__name__)

class Rewarder(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.cuda if self.args.gpu else "cpu")
        self.average_reward = 0.0

    def get_reward(self, predicted_e2, e2, r):
        rewards = []
        predicted_e2 = predicted_e2.cpu().numpy()
        e2 = e2.cpu().numpy()
        r = r.cpu().numpy()
        for i, pe2 in enumerate(predicted_e2):
            _r = r[i]
            _e2 = e2[i]
            if _e2 == pe2:
                reward = 1.0
            else:
                reward = -0.0
            rewards.append(reward)
        rewards = np.array(rewards)
        normed_reward = rewards - self.average_reward
        self.average_reward = self.args.beta * rewards.mean() + (1-self.args.beta) * self.average_reward
        normed_reward = (normed_reward - normed_reward.mean()) / (np.sqrt(normed_reward.var()) + 1e-5)
        return torch.tensor(normed_reward, dtype=float, device=self.device), rewards





class KBCompleter(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, KB, state_dict=None):
        # Book-keeping.
        self.args = args
        self.device = torch.device(args.cuda if self.args.gpu else "cpu")
        self.updates = 0
        self.use_cuda = False
        self.KB = KB
        self.rewarder = Rewarder(args)

        # Building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'minerva':
            self.network = Minerva(args, KB)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)
        if state_dict:
            self.network.load_state_dict(state_dict)


    def init_optimizer(self, state_dict=None):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: network parameters
        """

        self.parameters = [p for p in self.network.parameters() if p.requires_grad]

        if self.args.optimizer == 'adamax':
            self.optimizer = optim.Adam(self.parameters, lr=self.args.learning_rate, amsgrad = True)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def update(self, ex, epoch):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()

        # Transfer to GPU
        if self.use_cuda:
            e1, r, e2 = ex
            e1 = e1.to(self.device)
            r = r.to(self.device)
            e2 = e2.to(self.device)
        else:
            e1, r, e2 = ex

        # Run forward

        per_step_loss, chosen_er, policy_entropy = self.network(e1, r, e2, epoch)
        chosen_r = [h[0] for h in chosen_er]

        query_relations = r.cpu().numpy()

        if self.args.sample_check:
            reward_multiplier = []
            for i, rel in enumerate(query_relations):
                chosen = np.random.choice(len(self.KB.r_view[rel]), 10)
                sampled = [self.KB.r_view[rel][c] for c in chosen]
                sampled_e1 = [e[0] for e in sampled]
                sampled_e2 = [e[1] for e in sampled]
                ee = self.KB.follow_path(sampled_e1, [c[[i,]*10] for c in chosen_r])

                if ee == sampled_e2 or sum(ee) == 0:
                    reward_multiplier.append(1.0)
                else:
                    reward_multiplier.append(0.0)

        total_loss  = torch.stack(per_step_loss).squeeze(1)
        # if epoch == 15:
        #     import pdb
        #     pdb.set_trace()

        gamma = torch.tensor([self.args.gamma**p for p in range(self.args.num_steps-1, -1, -1)], device=self.device).view(self.args.num_steps,1)

        predicted_e2 = chosen_er[-1][-1]

        final_reward, R = self.rewarder.get_reward(predicted_e2, e2, r)

        if self.args.sample_check:
            reward_multiplier = torch.tensor(reward_multiplier, dtype=float, device=self.device)
            final_reward = final_reward * reward_multiplier

        reward = final_reward.repeat(self.args.num_steps).view(self.args.num_steps, -1) * gamma

        J = reward * total_loss

        policy_entropy = torch.stack(policy_entropy)

        loss = J.mean(1).sum(0) - self.args.Lambda * policy_entropy.mean()



        # Compute loss and accuracies


        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.args.grad_clip_norm)

        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(),
        #                                self.args.grad_clipping)

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        # self.reset_parameters()

        return final_reward.float().mean(), loss.item(), ex[0].size(0), R

    def reset_parameters(self):
        """Reset any partially fixed parameters to original states."""

        # Reset fixed embeddings to original value
        if self.args.tune_partial > 0:
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding

            # Embeddings to fix are the last indices
            offset = embedding.size(0) - fixed_embedding.size(0)
            if offset >= 0:
                embedding[offset:] = fixed_embedding

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, epoch, is_test=False, printer=None):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores
        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:
            e1, r, e2 = ex
            e1 = e1.to(self.device)
            r = r.to(self.device)
            e2 = e2.to(self.device)
        else:
            e1, r, e2 = ex

        # Run forward
        true_batch_size = e1.shape[0] // self.args.beam_size


        chosen_er, beam_log_probs = self.network(e1, r, e2, epoch=epoch)
        predicted_e2 = chosen_er[-1][-1]
        _, reward = self.rewarder.get_reward(predicted_e2, e2, r)


        if self.args.beam_search == 0:
            beam_log_probs = beam_log_probs.view( e1.shape[0], -1)
        if is_test:
            printer.save_paths(self.args, e1, r, e2, chosen_er, self.KB, beam_log_probs, print_reward=True, reward=reward)

        # print("batch")

        predicted_e = chosen_er[-1][-1]
        predicted_e = predicted_e.view(-1, self.args.beam_size).cpu().numpy()
        e2 = e2.view(-1, self.args.beam_size).cpu().numpy()



        return predicted_e, e2, chosen_er


    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def get_paths(self, e1, r, e2=None, epoch=None):
        """Forward a batch of examples only to get predictions.
        Args:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores
        If async_pool is given, these will be AsyncResult handles.
        """
        # Eval mode
        self.network.eval()

        # Transfer to GPU
        if self.use_cuda:

            e1 = e1.to(self.device)
            r = r.to(self.device)
            if e2 != None:
                e2 = e2.to(self.device)


        # Run forward
        true_batch_size = e1.shape[0] // self.args.beam_size

        chosen_er, beam_log_probs = self.network(e1, r, e2, epoch=epoch, end_point = False)

        if self.args.beam_search == 0:
            beam_log_probs = beam_log_probs.view(e1.shape[0], -1)


        return chosen_er, beam_log_probs




    def save(self, filename):

        network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'args': self.args,
            'KB':self.KB
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):

        network = self.network
        params = {
            'state_dict': network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True, new_KB=None, use_new_args=False):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        KB = saved_params['KB']
        if new_args and use_new_args:
            # if any new value is provided then use it
            for k,v in vars(new_args).items():
                if k not in vars(args):  # new arg, then copy it
                    logger.info("Adding a new arg {} with value {}".format(k, v))
                    vars(args)[k] = v
                if vars(args)[k] != vars(new_args)[k]:  # value dont match for the key; update with the new one.
                    logger.info("Setting the value of arg {} to {}. The value present in the loaded model was {}".format(k, v, vars(args)[k]))
                    vars(args)[k] = vars(new_args)[k]

        # for backward compatibility
        if new_args and not use_new_args:
            # if any new value is provided then use it
            for k,v in vars(new_args).items():
                if k not in vars(args):  # new arg, then copy it
                    logger.info("Adding a new arg {} with value {}".format(k, v))
                    vars(args)[k] = v


        return KBCompleter(args, KB, state_dict)

    # @staticmethod
    # def load_checkpoint(filename, normalize=True):
    #     logger.info('Loading model %s' % filename)
    #     saved_params = torch.load(
    #         filename, map_location=lambda storage, loc: storage
    #     )
    #     word_dict = saved_params['word_dict']
    #     feature_dict = saved_params['feature_dict']
    #     state_dict = saved_params['state_dict']
    #     epoch = saved_params['epoch']
    #     optimizer = saved_params['optimizer']
    #     args = saved_params['args']
    #     model = DocReader(args, word_dict, feature_dict, state_dict, normalize)
    #     model.init_optimizer(optimizer)
    #     return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.to(self.device)

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)