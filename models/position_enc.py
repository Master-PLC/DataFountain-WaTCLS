#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename     :position_enc.py
@Description  :
@Date         :2021/12/22 20:02:50
@Author       :Arctic Little Pig
@version      :1.0
'''

import math
from pdb import set_trace as stop

import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        # x shape: [batch_size, channels, height, width] = [32, 2048, 18, 18]
        # mask shape: [batch_size, height, width] = [32, 18, 18]

        # x = tensor_list.tensors
        # mask = tensor_list.mask
        assert mask is not None
        # Shape: [batch_size, height, width] = [32, 18, 18]
        not_mask = ~mask
        # stop()
        # 按维度累加
        y_embed = not_mask.cumsum(1)  # , dtype=torch.float32)
        x_embed = not_mask.cumsum(2)  # , dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # Shape: [batch_size, height, width] = [32, 18, 18]
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # , dtype=torch.float32)
        # Shape: [hidden/2] = [1024]
        dim_t = torch.arange(self.num_pos_feats, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Shape: [batch_size, height, width, hidden/2] = [32, 18, 18, 1024]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # stop()

        # Shape: [batch_size, height, width, hidden/4, 2] = [32, 18, 18, 512, 2]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4)
        # Shape: [batch_size, height, width, hidden/2] = [32, 18, 18, 1024]
        pos_x = pos_x.flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4)
        pos_y = pos_y.flatten(3)
        # Shape: [batch_size, hidden, height, width] = [32, 2048, 18, 18]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# def build_position_encoding(args):
#     N_steps = args.hidden_dim // 2
#     position_embedding = PositionEmbeddingSine(N_steps, normalize=True)


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            f"Cannot use sin/cos positional encoding with odd dimension (got dim={d_model:d})")
    # Shape: [d_model, height, width] = [2048, 18, 18]
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    # Shape: [d_model/4] = [512]
    div_term = torch.exp(torch.arange(0., d_model, 2)
                         * -(math.log(10000.0) / d_model))
    # Shape: [height, 1] = [18 ,1]
    pos_h = torch.arange(0., height).unsqueeze(1)
    # Shape: [width, 1] = [18 ,1]
    pos_w = torch.arange(0., width).unsqueeze(1)
    # Shape: [d_model/4, height, width] = [512 ,18, 18]
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(
        0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
