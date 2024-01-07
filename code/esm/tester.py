import os
import time
import numpy as np
from typing import List, Union, Tuple
from argparse import Namespace

import torch
from torch import Tensor, LongTensor

from model.xfmr_nhp_fast import XFMRNHPFast
from utils.metrics import distance_between_event_seq, count_mae
from esm.manager import Manager
from eval.sigtest import Bootstrapping

class Tester(Manager):

    def __init__(self, args, lggr, is_multistep: bool=False):
        tic = time.time()
        super(Tester, self).__init__(args, is_multistep)
        self.is_multistep = is_multistep
        if is_multistep:
            self.model.load_state_dict(torch.load(args.model_marks_dir))
            self.model_times = torch.load(args.model_times_dir, map_location=self.args.device)
        else:
            self.model.load_state_dict(torch.load(args.checkpoint))
        self.log = lggr
        self.log.info(f"Time spent on initialization: {time.time() - tic:.1f} secs")
        
    def macro_avg(self, x): 
        num, denom = 0.0, 0.0 
        for num_i, denom_i in x: 
            num += num_i
            denom += denom_i
        return num / denom
    
    def eval_distance(self, ground_truth_tuple, sample_tuple):
        distance = [distance_between_event_seq(ground_truth_tuple,
                                               sample_tuple,
                                               del_cost=[del_cost],
                                               trans_cost=1.0,
                                               num_types=self.test_loader_inv.dataset.event_num)[0][0] for del_cost in self.args.eval_del_cost]

        #return np.array(distance).mean()
        return distance
    
    def run_one_iteration_eval(self, model: XFMRNHPFast, dataLoader):
        model = model.eval()

        total_log_like = 0
        total_acc = 0
        num_tokens = 0
        pad_idx = self.train_loader.dataset.pad_index
        num_events = 0
        all_logs = []
        all_llkl_token = []
        all_type_errors = []
        with torch.inference_mode():
            for batch in dataLoader:
                new_batch = [x.cuda() for x in batch]
                time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = new_batch
                _neg_log_lkl, logits = model.compute_loglik(new_batch)
                _loss = torch.sum(_neg_log_lkl)
                total_log_like += -_loss.item()
                
                _neg_log_lkl = _neg_log_lkl[batch_non_pad_mask[:, 1:]]
                error_batch = torch.argmax(logits, dim=-1).ne(event_seq[:, 1:])[batch_non_pad_mask[:, 1:]]
    
                num_tokens += event_seq[:, 1:].ne(pad_idx).sum().item()
                num_events += (event_seq < pad_idx).sum().item()
                all_llkl_token.extend([(-x, 1.0) for x in _neg_log_lkl.tolist()])
                all_type_errors.extend([(x, 1.0) for x in error_batch.tolist()])
        
        assert len(all_llkl_token) == len(all_type_errors) and len(all_llkl_token) == num_tokens, \
            "The number of tokens are not the same! The list all_llkl_token has %d tokens while it should have %d tokens." %(len(all_llkl_token), num_tokens)
        
        return total_log_like, num_tokens, num_events, all_llkl_token, all_type_errors

    def eval_mae(self, ground_truth_tuple, sample_tuple):
        def most_frequent(list_):
            return max(set(list_), key=list_.count)

        most_freq_type = most_frequent(ground_truth_tuple[1])

        mae = count_mae(ground_truth_tuple,
                        sample_tuple,
                        most_freq_type)

        return mae

    def eval_acc(self, ground_truth_tuple, sample_tuple):
        event_label = ground_truth_tuple[1]
        event_pred = sample_tuple[1]
        min_len = min(len(event_pred), len(event_label))
        if len(event_pred) == 0 and len(event_label) != 0:
            return 0
        elif len(event_pred) != 0 and len(event_label) == 0:
            return 1
        return np.equal(event_pred[:min_len], event_label[:min_len]).mean()

    def eval_rmse(self, ground_truth_tuple, sample_tuple):
        time_label = ground_truth_tuple[0]
        time_pred = sample_tuple[0]
        return np.sqrt(np.mean(time_label - time_pred) ** 2)

    def filter_points(self, ground_truth_tuple, sample_tuple):
        time_range = self.args.time_range

        filter_max_time = ground_truth_tuple[0][0] + time_range
        horizon = len(ground_truth_tuple[0])

        def _truncate_tuple(one_tuple):
            end_i = horizon
            for i in range(horizon):
                if one_tuple[0][i] > filter_max_time:
                    end_i = i
                    break

            return one_tuple[0][:end_i], one_tuple[1][:end_i]

        # filter ground truth
        ground_truth_tuple = _truncate_tuple(ground_truth_tuple)
        sample_tuple = _truncate_tuple(sample_tuple)

        return ground_truth_tuple, sample_tuple

    def eval_l2_error(self, ground_truth_tuple, sample_tuple):
        label_type = ground_truth_tuple[1]
        pred_type = sample_tuple[1]

        def _type_count_vector(seq_type):
            vector = np.zeros(self.test_loader_inv.dataset.event_num)
            for k in seq_type:
                vector[k] = vector[k] + 1
            return vector

        label_vector = _type_count_vector(label_type)
        pred_vector = _type_count_vector(pred_type)

        # in the code i have not divided it by the num of event types.
        l2_error = np.sqrt(np.mean(np.square(label_vector - pred_vector)))
        return l2_error

    def evaluation_per_seq(self, ground_truth_tuple, sample_tuple):
        res = dict()

        ground_truth_tuple, sample_tuple = self.filter_points(ground_truth_tuple, sample_tuple)

        # distance evaluation
        distance = self.eval_distance(ground_truth_tuple, sample_tuple)
        res['joint_otd_avg'] = distance

        mae = self.eval_mae(ground_truth_tuple, sample_tuple)
        res['joint_mae'] = mae

        acc = self.eval_acc(ground_truth_tuple, sample_tuple)
        res['joint_acc'] = acc

        res['joint_rmse'] = self.eval_l2_error(ground_truth_tuple, sample_tuple)

        return res

    def eval_joint_distance(self, metrics_per_seq):
        seq_id = metrics_per_seq.keys()
        joint_distance = [metrics_per_seq[id]['joint_otd_avg'] for id in seq_id]
        joint_distance_avg_seq = np.array(joint_distance).mean(axis=0)

        return joint_distance_avg_seq

    def evaluation_average_by_seq(self, metrics_per_seq):
        seq_id = list(metrics_per_seq.keys())

        res_metrics = dict()
        desired_scalar_metrics = ['joint_acc', 'joint_rmse']
        for metric_name in desired_scalar_metrics:
            each_metric = [np.mean(metrics_per_seq[tmp_id][metric_name]) for tmp_id in seq_id]
            res_metrics[metric_name] = np.mean(each_metric)

        desired_vector_metrics = ['joint_otd_avg']
        for metric_name in desired_vector_metrics:
            dim = np.array(metrics_per_seq[seq_id[0]][metric_name]).shape[-1]
            tmp = np.array([val for tmp_id in seq_id for val in metrics_per_seq[tmp_id][metric_name]])
            tmp = tmp.reshape([-1, dim])
            res_metrics[metric_name] = tmp.mean(axis=0)

        return res_metrics
    
    def get_pred_dtimes(self, prev_marks: LongTensor) -> Tensor:
        return self.model_times[prev_marks.squeeze()]
    
    def run_generation_multi_step_Batch(self, batch: List[Tensor]) -> Union[Tuple[Tensor, Tensor], Tuple[None, None]]:
        """
        batch mode
        """       
        time_seqs, _, event_seqs, batch_non_pad_mask, batch_attention_mask, _ = batch
        mask_batch = batch_non_pad_mask.sum(1) > self.args.sample_len
        time_seqs, event_seqs, batch_non_pad_mask, batch_attention_mask = time_seqs[mask_batch], event_seqs[mask_batch], \
                                                                          batch_non_pad_mask[mask_batch], batch_attention_mask[mask_batch]
        batch_size = event_seqs.size(0)
        if batch_size == 0:
            return None, None
        
        times = [time_seqs[:, :-self.args.sample_len]]
        types = [event_seqs[:, :-self.args.sample_len]]
        for s in range(self.args.sample_len):
            time_seqs_history = torch.cat(times, axis=-1).to(self.args.device).squeeze() 
            event_seqs_history = torch.cat(types, axis=-1).to(self.args.device).squeeze()
            batch_non_pad_mask_history = batch_non_pad_mask[:, :(-self.args.sample_len + s)].to(self.args.device)
            batch_attention_mask_history = batch_attention_mask[:, :(-self.args.sample_len + s), :(-self.args.sample_len + s)].to(self.args.device)
            
            batch_history = [time_seqs_history, event_seqs_history, batch_non_pad_mask_history, batch_attention_mask_history]
            time_seqs_pred = (time_seqs_history[:, -1] + self.get_pred_dtimes(event_seqs_history[:, -1])).unsqueeze(-1) #pred_time[:, s].to(self.args.device).unsqueeze(-1)
            times.append(time_seqs_pred.cpu())
            
            logits_at_times = self.model.compute_logits_batch(batch_history, time_seqs_pred) # [batch_size, num_events]
            next_event_name = torch.argmax(logits_at_times, dim=-1).cpu() # [batch_size, 1] 
            types.append(next_event_name)
        
        return torch.cat(times, axis=-1).squeeze(), torch.cat(types, axis=-1).squeeze() # [batch_size, sequence_length]

    def run_one_epoch(self):
        sample_len = self.args.sample_len

        time_labels = []
        event_seq_labels = []
        time_preds = []
        type_preds = []
        self.log.info('Running multi-step prediction...')
        tic = time.time()
        for batch in self.test_loader_inv:
            time_seq, _, event_seq, batch_non_pad_mask, _, _ = batch
            
            time_pred_batch, type_pred_batch = self.run_generation_multi_step_Batch(batch)
            
            if time_pred_batch is None:
                    continue
            
            batch_size = time_seq.size(0)
            ii = 0
            for i in range(batch_size):
                pad_mask_i = batch_non_pad_mask[i]
                _time_seq, _event_seq = time_seq[i][pad_mask_i], event_seq[i][pad_mask_i]
                                                    
                # if sample_len >= seq_len, then we skip it
                if sample_len >= len(_time_seq):
                    continue
                
                time_pred, type_pred = time_pred_batch[ii][pad_mask_i], type_pred_batch[ii][pad_mask_i]
                ii += 1
                
                time_seq_label = _time_seq[-sample_len:].numpy().tolist()
                event_seq_label = _event_seq[-sample_len:].numpy().tolist()
                time_seq_label_pred = time_pred[-sample_len:].numpy().tolist()
                event_seq_label_pred = type_pred[-sample_len:].numpy().tolist()
                
                time_labels.append(time_seq_label)
                event_seq_labels.append(event_seq_label)
                time_preds.append(time_seq_label_pred)
                type_preds.append(event_seq_label_pred)
           
        time_alg = (time.time() - tic)
        message = 'Done! Prediction time: %.2f seconds' %time_alg
        self.log.info(message)
            
        metrics = None
        res = dict()
        seq_id = 0
        
        ground_truth_tuple_list = []
        pred_sample_tuple_list = []
        for seq_id, (time_seq_label, event_seq_label, time_pred, type_pred) in enumerate(zip(time_labels,
                                                                                event_seq_labels, time_preds,
                                                                                type_preds)):            
            # re-eval
            ground_truth_tuple = (time_seq_label, event_seq_label)

            # joint model tuple
            joint_tuple = (time_pred, type_pred)

            res[seq_id] = self.evaluation_per_seq(ground_truth_tuple, joint_tuple)

            ground_truth_tuple_list.append(ground_truth_tuple)
            pred_sample_tuple_list.append(joint_tuple)

        # min distance for each pred seq
        self.log.info("Num sequences: %d" %len(time_labels))
        metrics = self.evaluation_average_by_seq(res)
        for key, val in metrics.items():
            if type(val) in [np.ndarray, list, set]:
                #val = ["%.4f" % i for i in val]
                val = np.mean(val)
            message = f'\t{key}:\t{val: .4f}'
            self.log.info(message)

        return metrics
    
    def run(self):
        if self.is_multistep:
            with torch.inference_mode():
                self.model = self.model.eval()
                self.run_one_epoch()
        else:
            self.log.info("Start evaluation...")            
            tic = time.time()
            # testing with loglik
            total_log_like, total_num_token, total_num_event, log_lkl_tokens, error_tokens = \
                self.run_one_iteration_eval(self.model, self.test_loader)
            message = f"[Test] num_tokens: {total_num_token}, num_events: {total_num_event}\n"
            self.log.info(message)
            
            bs = Bootstrapping()
            for arr, name in zip([log_lkl_tokens, error_tokens],
                                [ "log-lkl", "error_rate"]):
                metric_mean = np.mean([x[0] for x in arr])
                metric_std = np.std([x[0] for x in arr])
                if name == "rmse":
                    metric_mean = np.sqrt(metric_mean)
                    metric_std = np.sqrt(metric_std)
                self.log.info(f"{name}: {metric_mean} ({metric_std})")
                bs.run(arr, self.log)
            self.log.info(f"Time spent on evaluation : {time.time() - tic:.2f}")
            self.log.info("\nEvaluation finished.")

