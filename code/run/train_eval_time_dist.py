import argparse
import time
import datetime
import os 
import pickle
import random
from typing import List, Tuple
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))

from eval.sigtest import Bootstrapping
from loguru import logger

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp

import torch

import warnings
warnings.filterwarnings("ignore")

from utils.misc import set_seed
        
        
def norm_logpdf(x, mu, sigma):
    log_exp_term = np.square((x - mu) / sigma)
    log_det = np.log(sigma)
    return -0.5 * (np.log(2 * np.pi) + log_exp_term) - log_det

def log_pdf_mixture_lognormals_stable(x, pi_vec, mu_vec, sigma_sq_vec):
    assert len(pi_vec) == len(mu_vec) and len(pi_vec) == len(sigma_sq_vec)
    assert np.isclose(np.sum(pi_vec), 1.) and (pi_vec >= 0.0).all() and (sigma_sq_vec > 0.0).all()
    assert (x >= 0.0).all()
    pi_vec[pi_vec <= 1e-8] = 1e-8

    if isinstance(x, np.ndarray):
        x[x <= 1e-8] = 1e-8
    else:
        x = 1e-8 if x <= 1e-8 else x
    
    sigma_vec = np.sqrt(sigma_sq_vec) 
    log_x = np.log(x)
    
    weighted_log_prob = np.zeros((len(x), len(pi_vec))) if isinstance(x, np.ndarray) else np.zeros((1, len(pi_vec)))
    for component in range(len(pi_vec)):
        weighted_log_prob[:, component] = norm_logpdf(log_x, mu_vec[component], sigma_vec[component])

    weighted_log_prob += np.log(pi_vec)

    return logsumexp(weighted_log_prob, axis=1) - log_x


def load_pkl(name, dict_name):
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        data = data[dict_name]
        
    return data, int(num_types)


def load_dataset(data_path: str="", dataset: str="so", split: str="train", folder: int=1, scale: float = 1.):
    assert split in ["train", "dev", "test"]
    assert folder > 0 and folder <= 5
    
    file_path = split + ".pkl"  
    if dataset in ["mimic", "so_old"]:
        file_path = os.path.join("fold%d" %folder, file_path)
        
    file_path = os.path.join(data_path, dataset, file_path)
    data, num_types = load_pkl(file_path, split)

    time = [[elem['time_since_start'] / scale for elem in inst] for inst in data]
    time_gap = [[elem['time_since_last_event'] / scale for elem in inst] for inst in data]
    event_type = [[elem['type_event'] for elem in inst] for inst in data]

    num_events = sum(len(seq) for seq in time)

    assert num_events == sum(len(seq) for seq in time_gap)
    assert num_events == sum(len(seq) for seq in event_type)

    return time, time_gap, event_type, num_types


def opt_logNormal_mixture(time_gap: List[np.ndarray], event_type: List[np.ndarray], num_types: int, num_components: int=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_gap = [tg[1:] for tg in time_gap]
    event_type = [et[:-1] for et in event_type]

    inter_arriv_full = np.concatenate(time_gap)
    inter_arriv_full[inter_arriv_full <= 1e-8] = 1e-8
    marks_inter_full = np.concatenate(event_type)

    mean_vec_opt = np.zeros(num_types)
    weights_opt = np.zeros((num_types, num_components))
    means_opt = np.zeros((num_types, num_components))
    variances_opt = np.ones((num_types, num_components))
    for u in range(num_types):
        data = inter_arriv_full[marks_inter_full == u][:, None]
        log_data = np.log(data)

        if len(data) > 5:
            model = GaussianMixture(n_components=num_components, n_init=1, init_params="k-means++", tol=1e-6, max_iter=30)
            model.fit(log_data)

            weights_opt[u] = model.weights_
            means_opt[u] = model.means_.flatten()
            variances_opt[u] = model.covariances_.flatten()
            mean_vec_opt[u] = (weights_opt[u] * np.exp(means_opt[u] + 0.5 * variances_opt[u])).sum()
        else:
            mean_vec_opt[u] = 1e-8
            weights_opt[u, 0] = 1.0
            means_opt[u, 0] = -8 * np.log(10)

    return mean_vec_opt, weights_opt, means_opt, variances_opt


def compute_scores(time_gap_seq: List[np.ndarray], event_type_seq: List[np.ndarray], mean_vec: np.ndarray) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    log_lkl_tokens, token_ses = [], []
    for s in range(len(event_type_seq)):
        marks = np.array(event_type_seq[s])
        time_gaps = np.array(time_gap_seq[s])
        time_gaps[time_gaps <= 1e-8] = 1e-8
         
        for idx in range(1, len(marks)):
            true_tau = time_gaps[idx]
            token_ses.append(((true_tau - mean_vec[marks[idx-1]])**2, 1.0))
        time_gaps = time_gaps[1:]
        marks = marks[:-1]
        for u in range(num_types):
            data = time_gaps[marks == u]
            for dd in log_pdf_mixture_lognormals_stable(data, weights_opt[u], means_opt[u], variances_opt[u]):
                log_lkl_tokens.append((dd, 1.0))

    assert len(token_ses) == len(log_lkl_tokens)
    return log_lkl_tokens, token_ses


def get_args():
    parser = argparse.ArgumentParser(description="Train the mixture of log-normals.")
    parser.add_argument('--dataset', type=str, required=True, choices=["mimic", "so", "taobao", "taxi", "so_old", "amazon"])
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the datasets.")
    parser.add_argument('--log_dir', type=str, required=True , help="Directory of logs.")
    parser.add_argument('--num_mixtures', type=int, default=2,  help='Number of mixture componenets.')
    parser.add_argument('--seed', default=2023, type=int, help='Set the random seed.')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.log_dir = os.path.join(args.log_dir, args.dataset)
    args.log_dir = os.path.join(args.log_dir, "time_dist")
    os.makedirs(args.log_dir, exist_ok=True)
    date_format = "%Y%m%d-%H%M%S"
    now_timestamp = datetime.datetime.now().strftime(date_format)
    filename_base = "MoG_numMixtures-{}_{}".format(args.num_mixtures, now_timestamp)
    logger.add(os.path.join(args.log_dir, f"log_{filename_base}.log"))
    set_seed(args.seed)
    
    time_train, time_gap_train, event_type_train, num_types = load_dataset(args.data_dir, args.dataset, split="train")
    time_dev, time_gap_dev, event_type_dev, _ = load_dataset(args.data_dir, args.dataset, split="dev")
    time_test, time_gap_test, event_type_test, _ = load_dataset(args.data_dir, args.dataset, split="test")

    num_events_train = sum(len(sq) for sq in time_train)
    num_events_dev = sum(len(sq) for sq in time_dev)
    num_events_test = sum(len(sq) for sq in time_test)

    dataset_full_name = 'Dataset: ' + args.dataset.upper()

    logger.info(20 * '-')
    logger.info(dataset_full_name)
    logger.info('num_types: %d\n' %num_types)
    
    logger.info('num_events train:      %d' %num_events_train)
    logger.info('num sequences train:   %d' %len(time_train))
    logger.info('num predictions train: %d\n' %(num_events_train - len(time_train)))
    
    logger.info('num_events dev:      %d' %num_events_dev)
    logger.info('num sequences dev:   %d' %len(time_dev))
    logger.info('num predictions dev: %d\n' %(num_events_dev - len(time_dev)))
    
    logger.info('num_events test:      %d' %num_events_test)
    logger.info('num sequences test:   %d' %len(time_test))
    logger.info('num predictions test: %d\n' %(num_events_test - len(time_test)))
    logger.info(20 * '-')
    logger.info("Start training...")
    
    start_t = time.time()
    log_first_gap_tr = np.log(np.array([time_gap_seq[0] + 1e-8 for time_gap_seq in time_gap_train]))
    mu_hat = log_first_gap_tr.mean()
    sigma_hat = log_first_gap_tr.std() + 1e-5

    mean_vec_opt, weights_opt, means_opt, variances_opt = opt_logNormal_mixture(time_gap_train, event_type_train, num_types=num_types, num_components=args.num_mixtures)       
    train_time = time.time() - start_t
    logger.info("Done!")
    
    saved_model_path = os.path.abspath(os.path.join(args.log_dir, 'saved_models'))
    os.makedirs(saved_model_path, exist_ok=True)
    logger.info("Saved models directory: {}".format(saved_model_path))
    torch.save(torch.from_numpy(mean_vec_opt), os.path.join(saved_model_path, "model_numMixtures-%d.pt" % args.num_mixtures))

    log_lkl_tokens_train, token_ses_train = compute_scores(time_gap_train, event_type_train, mean_vec_opt)
    log_lkl_tokens_dev, token_ses_dev = compute_scores(time_gap_dev, event_type_dev, mean_vec_opt)
    log_lkl_tokens_test, token_ses_test= compute_scores(time_gap_test, event_type_test, mean_vec_opt)
    
    logger.info("  Training time: %.1f min" %(train_time / 60.))
    bs = Bootstrapping()
    
    logger.info("\n\n" + 50 * " " + "***** Train Dataset Evaluation *****")
    logger.info("  Num predicted: %d" %len(log_lkl_tokens_train))
    for arr, name in zip([log_lkl_tokens_train, token_ses_train],
                         [ "log-lkl", "rmse"]):
        metric_mean = np.mean([x[0] for x in arr])
        metric_std = np.std([x[0] for x in arr])
        if name == "rmse":
            metric_mean = np.sqrt(metric_mean)
            metric_std = np.sqrt(metric_std)
        logger.info(f"{name}: {metric_mean} ({metric_std})")
        bs.run(arr, logger)
        
    logger.info("\n\n" + 50 * " " + "***** Dev Dataset Evaluation *****")
    logger.info("  Num predicted: %d" %len(log_lkl_tokens_dev))
    for arr, name in zip([log_lkl_tokens_dev, token_ses_dev],
                         [ "log-lkl", "rmse"]):
        metric_mean = np.mean([x[0] for x in arr])
        metric_std = np.std([x[0] for x in arr])
        if name == "rmse":
            metric_mean = np.sqrt(metric_mean)
            metric_std = np.sqrt(metric_std)
        logger.info(f"{name}: {metric_mean} ({metric_std})")
        bs.run(arr, logger)
        
    logger.info("\n\n" + 50 * " " + "***** Test Dataset Evaluation *****")
    logger.info("  Num predicted: %d" %len(log_lkl_tokens_test))
    for arr, name in zip([log_lkl_tokens_test, token_ses_test],
                         [ "log-lkl", "RMSE"]):
        metric_mean = np.mean([x[0] for x in arr])
        metric_std = np.std([x[0] for x in arr])
        if name == "rmse":
            metric_mean = np.sqrt(metric_mean)
            metric_std = np.sqrt(metric_std)
        logger.info(f"{name}: {metric_mean} ({metric_std})")
        bs.run(arr, logger)
