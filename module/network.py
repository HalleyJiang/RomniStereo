# File author: Hualie Jiang (jianghualie0@gmail.com)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from module.featurelayer import FeatureLayers, Conv2D
from module.volume_generator import Generator
from module.corr import CorrBlock1D
from module.update import UpdateBlock
from utils.common import *

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class ROmniStereo(torch.nn.Module):

    def __init__(self, varargin=None):
        super(ROmniStereo, self).__init__()
        opts = Edict()
        opts.use_rgb = False
        self.opts = argparse(opts, varargin)
        self.encoder = FeatureLayers(self.opts.base_channel, self.opts.use_rgb, self.opts.encoder_downsample_twice)
        context_dim = self.opts.base_channel
        hidden_dim = self.opts.base_channel*2
        self.volume_gen = Generator(self.opts)
        self.state_conv = Conv2D(context_dim, hidden_dim, 1, pad=0, relu=False)
        self.update_block = UpdateBlock(self.opts, hidden_dim=hidden_dim, input_dim=context_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def spherical_sweep(self, fisheye_feats, grids):
        bs = fisheye_feats[0].shape[0]
        grids_pad = [torch.cat([torch.zeros_like(grid[..., :1]), grid], dim=-1) for grid in grids]
        sph_feats = []
        for feat, grid in zip(fisheye_feats, grids_pad):
            sph_feat = F.grid_sample(feat[..., None], grid.repeat(bs, 1, 1, 1, 1), align_corners=True)
            sph_feats.append(sph_feat)

        # embed sampling girds
        sph_feats = sph_feats + [grid.permute(-1, 0, 1, 2).repeat(bs, 1, 1, 1, 1) for grid in grids]

        return sph_feats

    def upsample_invdepth_idx(self, invdepth, mask):
        """ Upsample invdepth field [H/2**n_ds, W/2**n_ds] -> [H, W] using convex combination """
        bs, ch, h, w = invdepth.shape
        factor = 2 ** self.opts.num_downsample
        mask = mask.view(bs, 1, 9, factor, factor, h, w)
        mask = torch.softmax(mask, dim=2)

        up_invdepth = F.unfold(factor * invdepth, [3, 3], padding=1)
        up_invdepth = up_invdepth.view(bs, ch, 9, 1, 1, h, w)

        up_invdepth = torch.sum(mask * up_invdepth, dim=2)
        up_invdepth = up_invdepth.permute(0, 1, 4, 2, 5, 3)
        return up_invdepth.reshape(bs, ch, factor*h, factor*w)

    def volume_sample(self, feat_volume, invdepth_idx):
        bs, ch, h, w, n_invd = feat_volume.shape

        invdepth_idx_floor = torch.floor(invdepth_idx)
        invdepth_idx_ceil = invdepth_idx_floor + 1
        invdepth_idx_floor = torch.clamp(invdepth_idx_floor, 0, n_invd-1)
        invdepth_idx_ceil = torch.clamp(invdepth_idx_ceil, 0, n_invd - 1)
        invdepth_idx = torch.clamp(invdepth_idx, 0, n_invd - 1)

        weight_floor = (invdepth_idx_ceil - invdepth_idx)
        weight_floor[weight_floor == n_invd - 1] = 1.0
        weight_ceil = (invdepth_idx - invdepth_idx_floor)
        weight_ceil[invdepth_idx_ceil == 0] = 1.0

        invdepth_idx_floor = invdepth_idx_floor.long()
        invdepth_idx_ceil = invdepth_idx_ceil.long()

        feat_floor = torch.gather(feat_volume, 4, invdepth_idx_floor.repeat(1, ch, 1, 1).unsqueeze(-1))[..., 0]
        feat_ceil = torch.gather(feat_volume, 4, invdepth_idx_ceil.repeat(1, ch, 1, 1).unsqueeze(-1))[..., 0]

        return weight_ceil*feat_ceil + weight_floor*feat_floor

    def forward(self, imgs, grids, iters=12, test_mode=False):
        with autocast(enabled=self.opts.mixed_precision):
            fisheye_feats = self.encoder(imgs)

        fisheye_feats = [feat.float() for feat in fisheye_feats]
        spherical_feats = self.spherical_sweep(fisheye_feats, grids)

        with autocast(enabled=self.opts.mixed_precision):
            match_feat_volume_list, context_feat_volume = self.volume_gen(spherical_feats)

        # initial context feature
        context_feat = context_feat_volume[..., 0]

        with autocast(enabled=self.opts.mixed_precision):
            inp = torch.relu(context_feat)
            net = torch.tanh(self.state_conv(context_feat))

        match_feat_volume_list = [feat.float() for feat in match_feat_volume_list]
        corr_fn = CorrBlock1D(*match_feat_volume_list,
                              radius=self.opts.corr_radius,
                              num_levels=self.opts.corr_levels)

        # initialize invdepth_idx
        invdepth_idx = torch.zeros_like(context_feat_volume[:, :1, ..., 0])

        invdepth_idx_predictions = []
        for itr in range(iters):
            invdepth_idx = invdepth_idx.detach()
            corr_feat = corr_fn(invdepth_idx)
            if itr > 0:
                context_feat = self.volume_sample(context_feat_volume, invdepth_idx)
                inp = torch.relu(context_feat)
            with autocast(enabled=self.opts.mixed_precision):
                net, delta_invdepth_idx, up_mask = self.update_block(net, inp, corr_feat, invdepth_idx,
                                                                     no_upsample=(test_mode and itr < iters - 1))
            invdepth_idx = invdepth_idx + delta_invdepth_idx

            if up_mask is not None:
                invdepth_idx_up = self.upsample_invdepth_idx(invdepth_idx, up_mask)
                invdepth_idx_predictions.append(invdepth_idx_up)

        if test_mode:
            return torch.clamp(invdepth_idx_predictions[-1], 0, self.opts.num_invdepth-1)

        return invdepth_idx_predictions
