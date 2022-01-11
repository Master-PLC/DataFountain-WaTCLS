#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :CTran.py
@Description  :
@Date         :2021/12/22 19:32:49
@Author       :Arctic Little Pig
@version      :1.0
'''

import numpy as np
import torch
import torch.nn as nn

from .backbone import Backbone
from .position_enc import PositionEmbeddingSine, positionalencoding2d
from .transformer_layers import SelfAttnLayer
from .utils import custom_replace, weights_init


class CTranModel(nn.Module):
    def __init__(self, num_labels, use_lmt, pos_emb=False, backbone='resnet101', layers=3, heads=4, dropout=0.1, no_x_features=False, downsample=False):
        super(CTranModel, self).__init__()
        self.use_lmt = use_lmt

        self.no_x_features = no_x_features  # (for no image features)

        # ResNet101 backbone
        self.backbone = Backbone(model_name=backbone)
        hidden = 2048  # this should match the backbone output feature size

        self.downsample = downsample
        if self.downsample:
            self.conv_downsample = nn.Conv2d(hidden, hidden, (1, 1))

        # Label Embeddings
        self.label_input = torch.Tensor(
            np.arange(num_labels)).view(1, -1).long()
        self.label_lt = nn.Embedding(
            num_labels, hidden, padding_idx=None)

        # State Embeddings
        # 索引0全部填充为 0，也即unknown的label的状态向量为0
        self.known_label_lt = nn.Embedding(3, hidden, padding_idx=0)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            self.position_encoding = PositionEmbeddingSine(
                int(hidden/2), normalize=True)
            # self.position_encoding = positionalencoding2d(
            #     hidden, 18, 18).unsqueeze(0)

        # Transformer
        self.self_attn_layers = nn.ModuleList(
            [SelfAttnLayer(hidden, heads, dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = nn.Linear(hidden, num_labels)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images, mask):
        # images shape: [batch_size, channels, height, width] = [32, 3, 576, 576]
        # mask shape: [batch_size, num_labels] = [32, 20]

        # Shape: [batch_size, num_labels] = [32, 20]
        const_label_input = self.label_input.repeat(images.size(0), 1).cuda()
        # Shape: [batch_size, num_labels, hidden] = [32, 20, 2048]
        init_label_embeddings = self.label_lt(const_label_input)

        # Shape: [batch_size, hidden, h5, w5] = [32, 2048, 18, 18]
        features = self.backbone(images)

        if self.downsample:
            # Shape: [batch_size, hidden, h5, w5] = [32, 2048, 18, 18]
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            # Shape: [batch_size, hidden, h5, w5] = [32, 2048, 18, 18]
            pos_encoding = self.position_encoding(features, torch.zeros(features.size(
                0), features.size(-2), features.size(-1), dtype=torch.bool).cuda())
            features = features + pos_encoding

        # Shape: [batch_size, h5*w5, hidden] = [32, 324, 2048]
        features = features.view(features.size(
            0), features.size(1), -1).permute(0, 2, 1)

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            # Shape: [batch_size, num_labels] = [32, 20]
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()

            # Get state embeddings
            # Shape: [batch_size, num_labels, hidden] = [32, 20, 2048]
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            # Shape: [batch_size, num_labels, hidden] = [32, 20, 2048]
            init_label_embeddings += state_embeddings

        if self.no_x_features:
            embeddings = init_label_embeddings
        else:
            # Concat image and label embeddings
            # Shape: [batch_size, h5*w5+num_labels, hidden] = [32, 344, 2048]
            embeddings = torch.cat((features, init_label_embeddings), 1)

        # Feed image and label embeddings through Transformer
        # Shape: [batch_size, h5*w5+num_labels, hidden] = [32, 344, 2048]
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            # Shape: [batch_size, h5*w5+num_labels, hidden] = [32, 344, 2048]
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        # Shape: [batch_size, num_labels, hidden] = [32, 20, 2048]
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        # Shape: [batch_size, num_labels, num_labels] = [32, 20, 20]
        output = self.output_linear(label_embeddings)

        # Shape: [1, num_labels, num_labels] = [1, 20, 20]
        diag_mask = torch.eye(output.size(1)).unsqueeze(0)
        # Shape: [batch_size, num_labels, num_labels] = [32, 20, 20]
        diag_mask = diag_mask.repeat(output.size(0), 1, 1).cuda()
        # Shape: [batch_size, num_labels] = [32, 20]
        output = (output * diag_mask).sum(-1)

        return output, None, attns
