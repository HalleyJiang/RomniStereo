# adapted from https://github.com/princeton-vl/RAFT-Stereo/blob/main/core/corr.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        bs, ch, h, w, nd = fmap1.shape
        self.bs = bs
        self.h = h
        self.w = w
        self.nd = nd
        corr = corr.reshape(bs*h*w, 1, 1, nd)

        self.corr_pyramid.append(corr)
        for i in range(1, self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)

    def __call__(self, invdepth_idx):
        r = self.radius
        coords = invdepth_idx.permute(0, 2, 3, 1)
        batch, h, w, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(2*r+1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch*h*w, 1, 1, 1) / 2**i
            x0 = 2*x0/(corr.shape[-1]-1) - 1
            y0 = torch.zeros_like(x0)
            coords_lvl = torch.cat([x0, y0], dim=-1)
            samp_corr = F.grid_sample(corr, coords_lvl, align_corners=True, mode='bilinear')
            samp_corr = samp_corr.view(batch, h, w, -1)
            out_pyramid.append(samp_corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        assert fmap1.shape == fmap2.shape
        bs, ch, h, w, nd = fmap1.shape
        corr = torch.einsum('aijkh,aijkh->ajkh', fmap1, fmap2)
        corr = corr.reshape(bs, h, w, 1, nd).contiguous()
        return corr / torch.sqrt(torch.tensor(ch).float())
