import os
import pickle

import torch
from tqdm import tqdm

from model.xfmr_nhp_fast import XFMRNHPFast
from data.data_loader import TPPDataset, create_dataloader
from data.NHPDataset import NHPDataset, createDataLoader
from utils.misc import set_seed
        
class Manager:
    def __init__(self, args, is_multistep: bool=False):
        set_seed(args.seed)
        self.args = args
        if is_multistep:
            self.test_loader, self.test_loader_inv = self.get_test_dataloaders()
            ds_input = self.test_loader.dataset
        else:
            [self.train_loader, self.dev_loader, self.test_loader] \
                = self.get_dataloader(args)
            ds_input = self.train_loader.dataset
            
        self.model = XFMRNHPFast(ds_input, args.dmodel, args.nLayers, args.nHeads, args.Dropout,
                                 args.TimeEmbeddingDim).to(self.args.device)
        

    def get_dataloader(self, args):
        loaders = []
        splits = ["train", 'dev', 'test']
        event_types = None
        token_types = 0
        file_path =  os.path.join(args.data_dir, args.dataset)  
        for _split in splits:
            with open(os.path.join(file_path, f"{_split}.pkl"), "rb") as f_in:
                # latin-1 for GaTech data
                try:
                    _data = pickle.load(f_in, encoding='latin-1')
                except:
                    _data = pickle.load(f_in)
                if event_types is None:
                    event_types = _data["dim_process"]
                else:
                    assert _data["dim_process"] == event_types, "inconsistent dim_process in different splits?"
                dataset = NHPDataset(_data[_split], event_types, concurrent=False, add_bos=False, add_eos=False)
                assert dataset.event_num <= event_types, f"{_split}.pkl has more event types than specified in dim_process!"
                token_types = max(token_types, dataset.num_types)
                loaders.append(createDataLoader(dataset, batch_size=args.BatchSize, shuffle=(_split == "train")))
        assert token_types > event_types, f"at least we should include [PAD]! token: {token_types}, event: {event_types}"
        return loaders
    
    def get_test_dataloaders(self):
        event_types = None
        token_types = 0
        _split = 'test'
            
        with open(os.path.join(self.args.data_dir, self.args.dataset, f"{_split}.pkl"), "rb") as f_in:
            # latin-1 for GaTech data
            try:
                _data = pickle.load(f_in, encoding='latin-1')
            except:
                _data = pickle.load(f_in)
            if event_types is None:
                event_types = _data["dim_process"]
            else:
                assert _data["dim_process"] == event_types, "inconsistent dim_process in different splits?"
            dataset = TPPDataset(_data[_split], event_types, concurrent=False, add_bos=False, add_eos=False,
                                    skip_padding=False)
            assert dataset.event_num <= event_types, f"{_split}.pkl has more event types than specified in dim_process!"
            token_types = max(token_types, dataset.num_types)
            dl_normal = create_dataloader(dataset, batch_size=self.args.BatchSize, shuffle=False)
            
            dataset_inverse = TPPDataset(_data[_split], event_types, concurrent=False, add_bos=False, add_eos=False,
                                    skip_padding=False, inverse=True)
            dl_inverse = create_dataloader(dataset_inverse, batch_size=self.args.BatchSize, shuffle=False)
        assert token_types > event_types, f"at least we should include [PAD]! token: {token_types}, event: {event_types}"
        return dl_normal, dl_inverse

    def run_one_iteration(self, model: XFMRNHPFast, dataLoader, mode, optimizer=None):
        assert mode in {"train", "eval"}
        if mode == "eval":
            model = model.eval()
        else:
            model = model.train()
            assert optimizer is not None
        total_log_like = 0
        total_acc = 0
        num_tokens = 0
        pad_idx = self.train_loader.dataset.pad_index
        num_events = 0
        for batch in dataLoader:
            new_batch = [x.cuda() for x in batch]
            time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = new_batch
            _neg_log_lkl, logits = model.compute_loglik(new_batch)
            _loss = torch.sum(_neg_log_lkl)
            if mode == "train":
                _loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_log_like += -_loss.item()
            
            total_acc += ((torch.argmax(logits, dim=-1) == event_seq[:, 1:]) * batch_non_pad_mask[:, 1:]).sum().item()
            num_tokens += event_seq[:, 1:].ne(pad_idx).sum().item()
            num_events += (event_seq < pad_idx).sum().item()

        return total_log_like, total_acc / num_tokens, num_tokens, num_events


