import argparse
import datetime
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))

from loguru import logger
import torch

from esm.trainer import Trainer
from esm.tester import Tester

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["mimic", "so", "taobao", "taxi", "so_old", "amazon"])
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the datasets.")
    parser.add_argument('--log_dir', type=str, required=True , help="Directory of logs.")
    
    # Training settings
    parser.add_argument('--MaxEpoch', default=100, type=int, help='max # training epochs')
    parser.add_argument('--BatchSize', default=64, type=int)
    parser.add_argument('--LearnRate', default=1e-4, type=float, help='learning rate')
    
    # Model parameters
    parser.add_argument('-teDim', '--TimeEmbeddingDim', default=128, type=int, help='the dimensionality of time embedding')
    parser.add_argument("--dmodel", type=int, default=64)
    parser.add_argument('--nLayers', default=2, type=int, help='the number of layers of Transformer')
    parser.add_argument("--nHeads", type=int, default=2)
    parser.add_argument('-dp', "--Dropout", type=float, default=0.1)
    
    # Other parameters
    parser.add_argument('--seed', default=2023, type=int, help='random seed')
    parser.add_argument('--cuda', type=int, default=0)    
    parser.add_argument('--IgnoreFirst', action='store_true', help='whether ignore the first interval in log-like computation?')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    args.log_dir = os.path.join(args.log_dir, args.dataset)
    args.log_dir = os.path.join(args.log_dir, "mark_dist")
    os.makedirs(args.log_dir, exist_ok=True)
    date_format = "%Y%m%d-%H%M%S"
    now_timestamp = datetime.datetime.now().strftime(date_format)
    filename_base = "Train_Eval_DTPP_marks_dmodel-{}_nLayers-{}_nHeads-{}_{}".format(args.dmodel, args.nLayers, args.nHeads, now_timestamp)
        
    logger.add(os.path.join(args.log_dir, f"log_{filename_base}.log"))
    logger.info(args)
    args.device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    logger.info("Training/Evaluating on device {}".format(args.device))

    trainer = Trainer(args, logger)
    trainer.run()
    args.checkpoint = trainer.args.checkpoint
    
    tester = Tester(args, logger)
    tester.run()

if __name__ == "__main__": 
    main()
