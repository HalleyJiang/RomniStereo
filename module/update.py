# adapted from https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/update.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
        super(DepthHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class MotionEncoder(nn.Module):
    def __init__(self, cor_planes, c1_planes=64, c2_planes=64, d1_planes=64, d2_planes=64, out_planes=128):
        super(MotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(cor_planes, c1_planes, 1, padding=0)
        self.convc2 = nn.Conv2d(c1_planes, c2_planes, 3, padding=1)
        self.convd1 = nn.Conv2d(1, d1_planes, 7, padding=3)
        self.convd2 = nn.Conv2d(d1_planes, d2_planes, 3, padding=1)
        self.conv = nn.Conv2d(c2_planes+d2_planes, out_planes-1, 3, padding=1)

    def forward(self, corr, invdepth):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        dep = F.relu(self.convd1(invdepth))
        dep = F.relu(self.convd2(dep))

        cor_dep = torch.cat([cor, dep], dim=1)
        out = F.relu(self.conv(cor_dep))
        return torch.cat([out, invdepth], dim=1)


class UpdateBlock(nn.Module):
    def __init__(self, opts, hidden_dim, input_dim):
        super(UpdateBlock, self).__init__()
        self.opts = opts
        self.encoder = MotionEncoder(opts.corr_levels * (2*opts.corr_radius + 1))
        encoder_output_dim = 128

        self.gru = ConvGRU(hidden_dim, encoder_output_dim+input_dim)

        self.depth_head = DepthHead(hidden_dim, hidden_dim=128, output_dim=1)
        factor = 2**self.opts.num_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, (factor**2)*9, 1, padding=0))

    def forward(self, net, inp, corr=None, inv_depth=None, no_upsample=False):
        motion_feat = self.encoder(corr, inv_depth)
        inp = torch.cat([inp, motion_feat], dim=1)
        net = self.gru(net, inp)

        delta_inv_depth = self.depth_head(net)

        if no_upsample:
            return net, delta_inv_depth, None

        # scale mask to balence gradients
        mask = .25 * self.mask(net)

        return net, delta_inv_depth, mask

