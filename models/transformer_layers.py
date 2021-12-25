#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :transformer_layers.py
@Description  :
@Date         :2021/12/22 20:20:41
@Author       :Arctic Little Pig
@version      :1.0
'''

import torch.nn as nn

from .utils import get_activation_fn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: [h5*w5+num_labels, batch_size, hidden] = [344, 32, 2048]

        src2, attn = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # src = src + self.dropout1(self.norm1(src2))
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(self.norm2(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super(SelfAttnLayer, self).__init__()
        self.transformer_layer = TransformerEncoderLayer(
            d_model, nhead, d_model*1, dropout=dropout, activation='relu')
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout=dropout, activation='gelu')

    def forward(self, k, mask=None):
        # k shape: [batch_size, h5*w5+num_labels, hidden] = [32, 344, 2048]
        attn = None
        # Shape: [h5*w5+num_labels, batch_size, hidden] = [344, 32, 2048]
        k = k.transpose(0, 1)
        # Shape: [h5*w5+num_labels, batch_size, hidden] = [344, 32, 2048]
        x, attn = self.transformer_layer(k, src_mask=mask)
        # Shape: [batch_size, h5*w5+num_labels, hidden] = [32, 344, 2048]
        x = x.transpose(0, 1)
        return x, attn
