import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
from typing import Optional, List
from torch import nn, Tensor

import pdb


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=1, dim_feedforward=2048,
                 activation="relu"):
        super().__init__()
        multihead_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        # FFN_conv = nn.Conv2d()  # do not use feed-forward network
        self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model,
                                          num_encoder_layers=num_layers)
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model,
                                          num_decoder_layers=num_layers)

    def forward(self, train_feat, test_feat, train_label):
        num_img_train = train_feat.shape[0]
        num_img_test = test_feat.shape[0]

        ## encoder
        encoded_memory, _ = self.encoder(train_feat, pos=None)

        ## decoder
        _, encoded_feat = self.decoder(train_feat.unsqueeze(0), memory=encoded_memory, pos=train_label,
                                           query_pos=None)

        _, decoded_feat = self.decoder(test_feat.unsqueeze(0), memory=encoded_memory, pos=train_label,
                                           query_pos=None)

        return encoded_feat, decoded_feat


class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def instance_norm(self, src, input_shape):
        batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(h, w, batch, dim).permute(2, 3, 0, 1)
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(batch, dim, -1).permute(2, 0, 1)
        src = src.reshape(-1, batch, dim)
        return src

    def forward(self, src, input_shape, pos: Optional[Tensor] = None):
        # query = key = value = src
        query = src
        key = src
        value = src

        # self-attention
        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2
        src = self.instance_norm(src, input_shape)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_encoder_layers=6, activation="relu"):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, pos: Optional[Tensor] = None):
        assert src.dim() == 4, 'Expect 5 dimensional inputs'
        src_shape = src.shape
        batch, dim, h, w = src.shape

        src = src.reshape(batch, dim, -1).permute(2, 0, 1)
        src = src.reshape(-1, batch, dim)

        if pos is not None:
            pos = pos.view(batch, 1, -1).permute(2, 0, 1)
            pos = pos.reshape(-1, batch, 1)

        output = src

        for layer in self.layers:
            output = layer(output, input_shape=src_shape, pos=pos)

        # [L,B,D] -> [B,D,L]
        output_feat = output.reshape(h, w, batch, dim).permute(2, 3, 0, 1)
        output_feat = output_feat.reshape(-1, dim, h, w)
        return output, output_feat


class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)

        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor * pos

    def instance_norm(self, src, input_shape):
        batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(h, w, batch, dim).permute(2, 3, 0, 1)
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(batch, dim, -1).permute(2, 0, 1)
        src = src.reshape(-1, batch, dim)
        return src

    def forward(self, tgt, memory, input_shape, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        # self-attention
        query = tgt
        key = tgt
        value = tgt

        tgt2 = self.self_attn(query=query, key=key, value=value)
        tgt = tgt + tgt2
        tgt = self.instance_norm(tgt, input_shape)

        mask = self.cross_attn(query=tgt, key=memory, value=pos)
        tgt2 = tgt * mask
        tgt2 = self.instance_norm(tgt2, input_shape)

        tgt3 = self.cross_attn(query=tgt, key=memory, value=memory * pos)
        tgt4 = tgt + tgt3
        tgt4 = self.instance_norm(tgt4, input_shape)

        tgt = tgt2 + tgt4
        tgt = self.instance_norm(tgt, input_shape)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_decoder_layers=6, activation="relu"):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)
        # self.post1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1)
        # self.activation = _get_activation_fn(activation)
        # self.post2 = nn.Conv2d(d_model, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, tgt, memory, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        assert tgt.dim() == 4, 'Expect 5 dimensional inputs'
        tgt_shape = tgt.shape
        batch, dim, h, w = tgt.shape

        if pos is not None:
            batch, h, w = pos.shape
            pos = pos.view(batch, 1, -1).permute(2, 0, 1)
            pos = pos.reshape(-1, batch, 1)
            pos = pos.repeat(1, 1, dim)

        tgt = tgt.view(batch, dim, -1).permute(2, 0, 1)
        tgt = tgt.reshape(-1, batch, dim)

        output = tgt

        for layer in self.layers:
            output = layer(output, memory, input_shape=tgt_shape, pos=pos, query_pos=query_pos)

        # [L,B,D] -> [B,D,L]
        output_feat = output.reshape(h, w, batch, dim).permute(2, 3, 0, 1)
        output_feat = output_feat.reshape(-1, dim, h, w)
        # output = self.post2(self.activation(self.post1(output)))
        return output, output_feat


def _get_clones(module, N):
    # return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    return nn.ModuleList([module for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))
        # self.out_conv = nn.Linear(n_head*key_feature_dim, feature_dim)  # bias=False

    def forward(self, query=None, key=None, value=None):
        isFirst = True
        for N in range(self.Nh):
            if (isFirst):
                concat = self.head[N](query, key, value)
                isFirst = False
            else:
                concat = torch.cat((concat, self.head[N](query, key, value)), -1)
        # output = self.out_conv(concat)
        output = concat
        return output


class RelationUnit(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.temp = 30
        self.WK = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        # self.WQ = nn.Linear(feature_dim, key_feature_dim)
        self.WV = nn.Linear(feature_dim, feature_dim)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        '''
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        '''

        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None):
        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1, 2, 0)  # Batch, Dim, Len_1

        w_q = self.WK(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1, 0, 2)  # Batch, Len_2, Dim

        dot_prod = torch.bmm(w_q, w_k)  # Batch, Len_2, Len_1
        affinity = F.softmax(dot_prod * self.temp, dim=-1)

        w_v = value.permute(1, 0, 2)  # Batch, Len_1, Dim
        output = torch.bmm(affinity, w_v)  # Batch, Len_2, Dim
        output = output.permute(1, 0, 2)

        return output


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """

    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                    torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3,
                              keepdim=True) + self.eps)).sqrt())  # view
        else:
            return input * (self.scale / (torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3,
                                                    keepdim=True) + self.eps).sqrt())
