

import argparse
import uuid
import os
from pprint import pprint
import uuid
import json

def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WN18RR", type=str)
    parser.add_argument("--max_num_actions", default=200, type=int)
    parser.add_argument("--hidden_size", default=100, type=int)
    parser.add_argument("--embedding_size", default=100, type=int) #Test with 100/200
    parser.add_argument("--l2_reg_const", default=1e-2, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float) #No found in TF
    parser.add_argument("--action_dropout", default=0, type=float) #No found in TF, seems not to be here
    parser.add_argument("--state_dropout", default=0.0, type=float)
    parser.add_argument("--beta", default=0.05, type=float) #WN = 0.05 FB = 0.02
    parser.add_argument("--positive_reward", default=1.0, type=float)
    parser.add_argument("--negative_reward", default=0, type=float)
    parser.add_argument("--log_dir", default="./logs/", type=str)
    parser.add_argument("--log_file_name", default="reward.txt", type=str)
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--num_rollouts", default=20, type=int)
    parser.add_argument("--beam_size", default=100, type=int) #test_rollout in TF
    parser.add_argument("--model_dir", default='', type=str)
    parser.add_argument("--base_output_dir", default='', type=str)
    parser.add_argument("--grad_clip_norm", default=5, type=int)
    parser.add_argument("--Lambda", default=0.05, type=float) #WN = 0.05 FB15 = 0.05
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--pool", default="max", type=str)
    parser.add_argument("--use_entity_embeddings", default=1, type=int)
    parser.add_argument("--use_neighbourhood_embeddings", default=0, type=int)
    parser.add_argument("--train_entity_embeddings", default=1, type=int)
    parser.add_argument("--sample_check", default=0, type=int)
    parser.add_argument("--beam_search", default=1, type=int)
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--log_file", default="log.txt", type=str)
    parser.add_argument("--model_type", default="minerva", type=str)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--cuda", default=1, type=int)
    parser.add_argument("--num_layers", default=1, type=int) #LSTM_layers in TF
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--data_workers", default=8, type=int)
    parser.add_argument("--num_steps", default=3, type=int)
    parser.add_argument("--cell_type", default="lstm", type=str)
    parser.add_argument("--optimizer", default="adamax", type=str)
    parser.add_argument("--eval_every", default=100, type=int)
    parser.add_argument("--display_iter", default=1000, type=int)
    parser.add_argument("--num_epochs", default=1000, type=int) #WN18 = 1000 FB15K = 2000
    parser.add_argument("--only_dev", default=0, type=int)
    parser.add_argument("--eval_on_train", default=0, type=int)
    parser.add_argument("--per_relation_scores", default=0, type=int)
    parser.add_argument("--print_paths", default=1, type=int)
    parser.add_argument("--load_model", default="", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--run_on_new_args", default=0, type=int)


    args = parser.parse_args()
    if args.use_entity_embeddings and args.use_neighbourhood_embeddings:
        raise ValueError("Only one of args.use_neighbourhood_embeddings and args.use_neighbourhood_embeddings"
                         " can be used")

    args.train = os.path.join("datasets/", args.dataset, "train.txt")
    args.graph = os.path.join("datasets/", args.dataset, "graph.txt")
    args.dev = os.path.join("datasets/", args.dataset, "train.txt")
    if args.only_dev == 1:
        args.test = os.path.join("datasets/", args.dataset, "train.txt")
    else:
        args.test = os.path.join("datasets/", args.dataset, "test.txt")

    if args.eval_on_train == 1:
        args.test = args.train
        args.dev = args.train

    args.page_rank = os.path.join("datasets/", args.dataset, "raw.pgrk") if os.path.exists(os.path.join("datasets/", args.dataset, "raw.pgrk")) else None
    args.page_rank = None
    args.output_dir = args.output_dir+"/"+args.dataset+"/"+str(uuid.uuid4())[:4]
    os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "args.json" ), "w") as args_out:
        json.dump(args.__dict__, args_out)

    return  args
