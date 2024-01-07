from typing import Tuple, List, Optional
import math

import torch
from torch import nn, Tensor

from model.xfmr import EncoderLayer, MultiHeadAttention

class XFMRNHPFast(nn.Module):
    def __init__(self, dataset, d_model, n_layers, n_head, dropout, d_time, use_norm=False,
                 sharing_param_layer=False):
        super(XFMRNHPFast, self).__init__()
        self.d_model = d_model
        self.d_time = d_time

        self.div_term = torch.exp(torch.arange(0, d_time, 2) * -(math.log(10000.0) / d_time)).reshape(1, 1, -1)
        # here num_types already includes [PAD], [BOS], [EOS]
        self.Emb = nn.Embedding(dataset.num_types, d_model, padding_idx=dataset.pad_index)
        
        self.n_layers = n_layers
        self.n_head = n_head
        self.sharing_param_layer = sharing_param_layer
        if not sharing_param_layer:
            self.heads = []
            for i in range(n_head):
                self.heads.append(
                    nn.ModuleList(
                        [EncoderLayer(
                            d_model + d_time,
                            MultiHeadAttention(1, d_model + d_time, d_model, dropout, output_linear=False),
                            use_residual=False,
                            dropout=dropout
                        )
                            for _ in range(n_layers)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        else:
            self.heads = []
            for i in range(n_head):
                self.heads.append(
                    nn.ModuleList(
                        [EncoderLayer(
                            d_model + d_time,
                            MultiHeadAttention(1, d_model + d_time, d_model, dropout, output_linear=False),
                            use_residual=False,
                            dropout=dropout
                        )
                            for _ in range(0)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model)
            
        self.type_predictor = nn.Linear(d_model * n_head, dataset.event_num, bias=True)
        nn.init.xavier_normal_(self.type_predictor.weight)
        
        self.xe_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=dataset.event_num)

    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        return pe

    def forward_pass(self, init_cur_layer_: Tensor, tem_enc: Tensor, 
                     tem_enc_layer: Tensor, enc_input: Tensor, 
                     combined_mask: Tensor, batch_non_pad_mask: Optional[Tensor]=None) -> Tensor:
        """
        init_cur_layer_: [batch_size, num_predicted, hidden_dim]
        tem_enc: [batch_size, history, time_emb_dim]
        tem_enc_layer: [batch_size, num_predicted, time_emb_dim]
        enc_input: [batch_size, history, hidden_dim + time_emb_dim]
        combined_mask: [batch_size, 2*seq_length, , 2*seq_length] (for training) or [batch_size, seq_length+1, , seq_length+1]
        batch_non_pad_mask: if not None then [batch_size, seq_length]
        """
        cur_layers = []
        seq_len = enc_input.size(1)
        for head_i in range(self.n_head):
            cur_layer_ = init_cur_layer_
            for layer_i in range(self.n_layers):
                layer_ = torch.cat([cur_layer_, tem_enc_layer], dim=-1)
                _combined_input = torch.cat([enc_input, layer_], dim=1)
                if self.sharing_param_layer:
                    enc_layer = self.heads[head_i][0]
                else:
                    enc_layer = self.heads[head_i][layer_i]
                enc_output = enc_layer(
                    _combined_input,
                    combined_mask
                )
                if batch_non_pad_mask is not None:
                    _cur_layer_ = enc_output[:, seq_len:, :] * (batch_non_pad_mask.unsqueeze(-1))
                else:
                    _cur_layer_ = enc_output[:, seq_len:, :]

                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_
                enc_input = torch.cat([enc_output[:, :seq_len, :], tem_enc], dim=-1)

                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    def forward(self, event_seqs: Tensor, time_seqs: Tensor, 
                batch_non_pad_mask: Tensor, attention_mask: Tensor, extra_times: Optional[Tensor]=None) -> Tensor:
        tem_enc = self.compute_temporal_embedding(time_seqs)
        tem_enc *= batch_non_pad_mask.unsqueeze(-1)
        enc_input = torch.tanh(self.Emb(event_seqs))
        init_cur_layer_ = torch.zeros_like(enc_input)
        layer_mask = (torch.eye(attention_mask.size(1)) < 1).unsqueeze(0).expand_as(attention_mask).to(
            attention_mask.device)
        if extra_times is None:
            tem_enc_layer = tem_enc
        else:
            tem_enc_layer = self.compute_temporal_embedding(extra_times)
            tem_enc_layer *= batch_non_pad_mask.unsqueeze(-1)
        # batch_size * (seq_len) * (2 * seq_len)
        _combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # batch_size * (2 * seq_len) * (2 * seq_len)
        contextual_mask = torch.cat([attention_mask, torch.ones_like(layer_mask)], dim=-1)
        _combined_mask = torch.cat([contextual_mask, _combined_mask], dim=1)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, tem_enc_layer, enc_input, _combined_mask, batch_non_pad_mask)

        return cur_layer_

    def compute_loglik(self, batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = batch
        
        enc_out = self.forward(event_seq[:, :-1], time_seq[:, :-1], batch_non_pad_mask[:, 1:], attention_mask[:, 1:, :-1], time_seq[:, 1:])
        logits = self.type_predictor(enc_out)
        loss = self.xe_loss(logits.transpose(1, 2), event_seq[:, 1:])

        return loss, logits
    
    def compute_logits_batch(self, batch: List[Tensor], pred_times_next: Tensor) -> Tensor:
        """
        Input:
        event_seqs, time_seqs: [batch_size, seq_length]
        pred_times_next, length_seqs: [batch_size, 1]
        Output: [batch_size, num_events]
        """
        
        time_seqs, event_seqs, batch_non_pad_mask, attention_mask = batch

        batch_size = event_seqs.size(0)
        assert (time_seqs[:, -1:] <= pred_times_next).all(), "predicted times must occur not earlier than last events!"
        
        tem_enc = self.compute_temporal_embedding(time_seqs)
        tem_enc *= batch_non_pad_mask.unsqueeze(-1)
        enc_input = torch.tanh(self.Emb(event_seqs))
        
        init_cur_layer_ = torch.zeros((batch_size, 1, enc_input.size(-1))).to(event_seqs.device)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        tem_layer_ = self.compute_temporal_embedding(pred_times_next)
        
        enc_out = self.forward_pass(init_cur_layer_, tem_enc, tem_layer_, enc_input, attention_mask)
        logits = self.type_predictor(enc_out)
        
        return logits