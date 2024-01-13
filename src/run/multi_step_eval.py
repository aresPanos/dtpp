import argparse
import os

import sys

from os import path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))

from loguru import logger
import torch

from esm.tester import Tester

TIME_RANGES = {"taxi": 4.5, "taobao": 1.5, "so": 20.}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='taxi', choices=["taobao", "taxi", "so"],   help = 'dataset')
    
    # Training settings
    parser.add_argument('--BatchSize', default=8, type=int)
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the datasets.")
    parser.add_argument('--log_dir', type=str, required=True , help="Directory of logs.")
    
    # Evaluation settings time_range
    parser.add_argument('--sample_len', default=20, type=int)
    parser.add_argument('--eval_del_cost', default=[0.05, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], type=list)
    
    # Marks Model parameters - These parameters should agree with the parameters of the trained model at 'model_marks_dir'
    parser.add_argument('-teDim', '--TimeEmbeddingDim', default=128, type=int, help='the dimensionality of time embedding')
    parser.add_argument("--dmodel", type=int, default=64)
    parser.add_argument('--nLayers', default=2, type=int, help='the number of layers of Transformer')
    parser.add_argument("--nHeads", type=int, default=2)
    parser.add_argument('-dp', "--Dropout", type=float, default=0.1)
    
    
    # Other parameters
    parser.add_argument('--seed', default=2023, type=int, help='random seed')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--model_times_dir', type=str, required=True, help="Directory of the trained the time model.")
    parser.add_argument('--model_marks_dir', type=str, required=True, help="Directory of the trained the mark model.")
    args = parser.parse_args()
    args.time_range = TIME_RANGES[args.dataset]
    return args

def main():
    args = get_args()
    args.log_dir = os.path.join(args.log_dir, args.dataset)
    args.log_dir_multi = os.path.join(args.log_dir, "multistep_pred")
    os.makedirs(args.log_dir_multi, exist_ok=True)
    num_mixtures = int(args.model_times_dir.split("model_numMixtures-")[-1][:-3])
    filename_base = "Eval_dmodel-{}_nLayers-{}_nHeads-{}_Times_nMixt-{}".format(args.dmodel, args.nLayers, args.nHeads, num_mixtures)
    
    logger.add(os.path.join(args.log_dir_multi, f"log_{filename_base}.txt"))
    
    logger.info(args)
    args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    logger.info("Evaluating on device {}".format(args.device))

    tester = Tester(args, logger, is_multistep=True)
    tester.run()

if __name__ == "__main__": 
    main()
