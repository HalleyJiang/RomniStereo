# File author: Hualie Jiang (jianghualie0@gmail.com)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, ch_in, ch_hid, ch_out=1):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Conv3d(ch_in, ch_hid, (1, 1, 1))
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Conv3d(ch_hid, ch_out, (1, 1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.out_act(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        ch_in = opts.base_channel

        self.reference_mapping = MLP(2*ch_in+4, ch_in)
        self.target_mapping = MLP(2*ch_in+4, ch_in)

    def forward(self, spherical_feats):

        front_weight = self.reference_mapping(torch.cat(spherical_feats[0::2], dim=1))
        reference_feat = front_weight*spherical_feats[0] + (1-front_weight)*spherical_feats[2]

        right_weight = self.target_mapping(torch.cat(spherical_feats[1::2], dim=1))
        target_feat = right_weight * spherical_feats[1] + (1 - right_weight) * spherical_feats[3]

        context_feat = reference_feat

        return [reference_feat, target_feat], context_feat
