import time
import datetime
import pickle
import os

import torch
from torch import optim

from utils.log import LogModelVal
from utils.misc import set_seed
from esm.manager import Manager

class Trainer(Manager):

    def __init__(self, args, lggr):
        set_seed(args.seed)
        super(Trainer, self).__init__(args)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.LearnRate
        )
        self.optimizer.zero_grad() # init clear
        self.log = lggr
        self.lggr_val = LogModelVal()
        self.lggr_val.initBest()

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log.info("the number of trainable parameters: " + str(num_params))
        
        self.saved_model_path = os.path.abspath(os.path.join(args.log_dir, 'saved_models'))
        os.makedirs(self.saved_model_path, exist_ok=True)
        self.log.info("Saved models directory: {}".format(self.saved_model_path))
        
        date_format = "%Y%m%d-%H%M%S"
        now_timestamp = datetime.datetime.now().strftime(date_format)
        
        model_name = self.model.__class__.__name__
        filename_base = "{}_dmodel-{}_nLayers-{}_nHeads-{}_{}".format(model_name, self.args.dmodel, self.args.nLayers, self.args.nHeads, now_timestamp)
        filename_model_save = filename_base + ".pt"
        self.args.checkpoint = os.path.join(self.saved_model_path, filename_model_save)

    def run(self):
        self.log.info("Start training...")
        for _epoch in range(self.args.MaxEpoch):
            tic = time.time()
            log_lik, acc, num_tokens, num_events = self.run_one_iteration(self.model, self.train_loader, "train", self.optimizer)
            time_train = (time.time() - tic)
            message = f"[ Epoch {_epoch} (train) ]: time to train one epoch is {time_train: .2f}, train log-like is {log_lik / num_tokens: .4f}, num_tokens: {num_tokens}, num_events: {num_events}, " \
                      f"acc is {acc : .4f} "
            self.log.info(message)
            with torch.no_grad():
                tic = time.time()
                log_lik, acc, num_tokens, num_events = self.run_one_iteration(self.model, self.dev_loader, "eval")
                time_valid = (time.time() - tic)
                message = f"[ Epoch {_epoch} (valid) ]: time to validate is {time_valid:.4f}, valid log-like is {log_lik / num_tokens: .4f}, num_tokens: {num_tokens}, num_events: {num_events}, " \
                      f", valid acc is {acc : .4f} "
                self.log.info(message)
                updated = self.lggr_val.updateBest("loglik", log_lik / num_tokens, _epoch)
                message = "current best loglik is {:.4f} (updated at epoch-{})".format(
                    self.lggr_val.current_best['loglik'], self.lggr_val.episode_best)
                if updated:
                    message += f", best updated at this epoch"
                    torch.save(self.model.state_dict(), self.args.checkpoint)

                self.log.info(message)
        self.log.info("End training!\n")



