# File author: Hualie Jiang (jianghualie0@gmail.com)

from __future__ import print_function, division

from argparse import ArgumentParser
import logging
import time
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Torch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# Internal modules
from dataset import Dataset, MultiDataset
from utils.common import *
from utils.image import *
from module.network import ROmniStereo

# Initialize
torch.backends.cudnn.benchmark = True
torch.backends.cuda.benchmark = True

parser = ArgumentParser(description='Evaluation for ROmniStereo')
parser.add_argument('--name', default='ROmniStereo', help="name of your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")

parser.add_argument('--db_root', default='../omnidata', type=str, help='path to dataset')
parser.add_argument('--dbname', default='itbt_sample', type=str,
                    choices=['itbt_sample', 'real_indoor_sample'], help='databases to test')

# data options
parser.add_argument('--phi_deg', type=float, default=45.0, help='phi_deg')
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640], help="size of out ERP.")

parser.add_argument('--valid_iters', type=int, default=12,
                    help='number of flow-field updates during validation forward pass')

parser.add_argument('--vis', action='store_true', help='oneline visualization')
parser.add_argument('--save_result', action='store_true', help='save inverse depth prediction results')
parser.add_argument('--save_misc', action='store_true', help='save misc')
parser.add_argument('--save_point_cloud', action='store_true', help='save point cloud')

args = parser.parse_args()

opts = Edict()
opts.snapshot_path = args.restore_ckpt
opts.name = args.name

opts.dbname = args.dbname
opts.db_root = args.db_root

opts.data_opts = Edict()
opts.data_opts.color_aug = False
opts.data_opts.phi_deg = args.phi_deg
opts.data_opts.equirect_size = args.equirect_size

opts.valid_iters = args.valid_iters
opts.net_opts = Edict()

# Results
opts.vis = args.vis
opts.save_result = args.save_result
opts.save_misc = args.save_misc
opts.save_point_cloud = args.save_point_cloud
snapshot_name = osp.splitext(osp.basename(opts.snapshot_path))[0]
opts.result_dir = osp.join('../results', opts.dbname)
opts.out_invdepth_fmt = osp.join(opts.result_dir, 'invdepth_%05d_'+snapshot_name+'.tiff')
opts.out_misc_fmt = osp.join(opts.result_dir, 'misc_%05d_'+snapshot_name+'.png')
opts.out_point_fmt = osp.join(opts.result_dir, 'pc_%05d_'+snapshot_name+'.ply')

if opts.vis:
    fig = plt.figure(frameon=False, figsize=(25, 10), dpi=40)
    plt.ion()
    plt.show()


def main():
    if not osp.exists(opts.snapshot_path):
        sys.exit('%s does not exsits' % (opts.snapshot_path))
    snapshot = torch.load(opts.snapshot_path)

    opts.net_opts = snapshot['net_opts']
    net = torch.nn.DataParallel(ROmniStereo(opts.net_opts), device_ids=[0])
    net.load_state_dict(snapshot['net_state_dict'])
    opts.data_opts.use_rgb = opts.net_opts.use_rgb
    opts.data_opts.num_invdepth = opts.net_opts.num_invdepth
    opts.data_opts.num_downsample = opts.net_opts.num_downsample

    data = Dataset(opts.dbname, opts.data_opts, db_root=opts.db_root, train=False)

    grids = [torch.tensor(grid, requires_grad=False).cuda() for grid in data.grids]

    if not osp.exists(opts.result_dir):
        os.makedirs(opts.result_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.result_dir))

    for d in range(data.data_size):
        fidx = data.frame_idx[d]
        imgs, gt, valid, raw_imgs = data.loadSample(fidx)
        net.eval()
        imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]
        with torch.no_grad():
            invdepth_idx = net(imgs, grids, opts.valid_iters, test_mode=True)
        invdepth_idx = toNumpy(invdepth_idx[0, 0])
        invdepth = data.indexToInvdepth(invdepth_idx)

        # Visualization
        if opts.vis or opts.save_misc or opt.save_point_cloud:
            vis_img, inputs_rgb, pano_rgb, invdepth_rgb, _ = data.makeVisImage(raw_imgs, invdepth, gt, return_all=True)
            if opts.vis:
                fig.clf()
                plt.imshow(vis_img)
                plt.axis('off')
                plt.tight_layout()
                plt.draw()
                plt.pause(0.5)
            if opts.save_misc:
                writeImage(vis_img, opts.out_misc_fmt % fidx)
                writeImage(inputs_rgb, opts.out_misc_fmt.replace('misc', 'input') % fidx)
                writeImage(pano_rgb, opts.out_misc_fmt.replace('misc', 'pano') % fidx)
                writeImage(invdepth_rgb, opts.out_misc_fmt.replace('misc', 'idepth') % fidx)
            if opts.save_point_cloud:
                data.writePointCloud(pano_rgb, invdepth,
                                     opts.out_point_fmt % fidx)

        # Save result
        if opts.save_result:
            data.writeInvdepth(invdepth,
                               opts.out_invdepth_fmt % fidx)


if __name__ == "__main__":
    main()
