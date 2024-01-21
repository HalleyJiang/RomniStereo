# File author: Hualie Jiang (jianghualie0@gmail.com)

from __future__ import print_function, division

import time
import random
from argparse import ArgumentParser

# Torch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# Internal modules
from dataset import Dataset, MultiDataset
from utils.common import *
from utils.image import *
from module.network import ROmniStereo
from module.loss_functions import sequence_loss

# Initialize
torch.backends.cudnn.benchmark = True
torch.backends.cuda.benchmark = True

parser = ArgumentParser(description='Training for ROmniStereo')

parser.add_argument('--name', default='ROmniStereo', help="name of your experiment")
parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--pretrain_ckpt', help="pretrained checkpoint for finetuning")

parser.add_argument('--db_root', default='../omnidata', type=str, help='path to dataset')
parser.add_argument('--dbname', nargs='+', default=['omnithings'], type=str,
                    choices=['omnithings', 'omnihouse', 'sunny', 'cloudy', 'sunset'],  help='databases to train')

# data options
parser.add_argument('--phi_deg', type=float, default=45.0, help='phi_deg')
parser.add_argument('--num_invdepth', type=int, default=192, metavar='N', help='number of disparity')
parser.add_argument('--equirect_size', type=int, nargs='+', default=[160, 640], help="size of out ERP.")
parser.add_argument('--use_rgb', action='store_true', help='use 3-channel rgb color images as input')

# net options
parser.add_argument('--base_channel', type=int, default=4, help='base channel of the network')
parser.add_argument('--encoder_downsample_twice', action='store_true',
                    help='the feature extractor downsample the fisheye input twice instead once.')
parser.add_argument('--num_downsample', type=int, default=1, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")

parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--fix_bn', action='store_true', help='fix batch normalization')

# training options
parser.add_argument('--total_epochs', type=int, default=30, help='total epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--train_iters', type=int, default=12,
                    help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--valid_iters', type=int, default=12,
                    help='number of flow-field updates during validation forward pass')
parser.add_argument('--lr', type=float, default=0.0005, help="max learning rate.")
parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

args = parser.parse_args()

opts = Edict()
# Dataset & sweep arguments
opts.name = args.name
opts.model_dir = os.path.join('./checkpoints', args.name)
opts.runs_dir = os.path.join('./runs', args.name)

opts.snapshot_path = args.restore_ckpt
opts.pretrain_path = args.pretrain_ckpt
opts.dbname = args.dbname
opts.db_root = args.db_root

opts.data_opts = Edict()
opts.data_opts.phi_deg = args.phi_deg
opts.data_opts.num_invdepth = args.num_invdepth
opts.data_opts.equirect_size = args.equirect_size
opts.data_opts.num_downsample = args.num_downsample
opts.data_opts.use_rgb = args.use_rgb

opts.net_opts = Edict()
opts.net_opts.base_channel = args.base_channel
opts.net_opts.num_invdepth = opts.data_opts.num_invdepth
opts.net_opts.use_rgb = opts.data_opts.use_rgb
opts.net_opts.encoder_downsample_twice = args.encoder_downsample_twice
opts.net_opts.num_downsample = args.num_downsample
opts.net_opts.corr_levels = args.corr_levels
opts.net_opts.corr_radius = args.corr_radius
opts.net_opts.mixed_precision = args.mixed_precision
opts.net_opts.fix_bn = args.fix_bn

opts.total_epochs = args.total_epochs
opts.batch_size = args.batch_size
opts.train_iters = args.train_iters
opts.valid_iters = args.valid_iters
opts.lr = args.lr
opts.wdecay = args.wdecay


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(model, num_steps):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, opts.lr, num_steps+100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def train(epoch_total, load_state):
    if len(opts.dbname) > 1:
        data = MultiDataset(opts.dbname, opts.data_opts, db_root=opts.db_root)
    else:
        data = Dataset(opts.dbname[0], opts.data_opts, db_root=opts.db_root)
    dbloader = torch.utils.data.DataLoader(data, batch_size=opts.batch_size,
                                           pin_memory=True, shuffle=True,
                                           num_workers=0, drop_last=True)
    total_num_steps = len(data)*opts.total_epochs//opts.batch_size

    net = nn.DataParallel(ROmniStereo(opts.net_opts)).cuda()
    if opts.net_opts.fix_bn:
        net.module.freeze_bn()
    LOG_INFO("Parameter Count: %d" % count_parameters(net))

    optimizer, scheduler = fetch_optimizer(net, total_num_steps)
    scaler = GradScaler(enabled=opts.net_opts.mixed_precision)

    start_epoch = 0
    if load_state:
        if opts.snapshot_path and osp.exists(opts.snapshot_path):
            snapshot = torch.load(opts.snapshot_path)
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.snapshot_path))
            if 'epoch' in snapshot.keys():
                start_epoch = snapshot['epoch'] + 1
            if 'epoch_loss' in snapshot.keys():
                epoch_loss = snapshot['epoch_loss']
            if 'optimizer' in snapshot.keys():
                optimizer.load_state_dict(snapshot['optimizer'])
            if 'epoch' in snapshot.keys() and 'epoch_loss' in snapshot.keys():
                LOG_INFO('startepoch:%d epoch_loss:%f' % (start_epoch, epoch_loss))
        elif opts.pretrain_path is None:
            sys.exit('%s do not exsits' % (opts.snapshot_path))

        if opts.pretrain_path and osp.exists(opts.pretrain_path):
            snapshot = torch.load(opts.pretrain_path)
            if 'net_state_dict' in snapshot.keys():
                net.load_state_dict(snapshot['net_state_dict'])
                LOG_INFO('checkpoint %s is loaded' % (opts.pretrain_path))
        elif opts.snapshot_path is None:
            sys.exit('%s do not exsits' % (opts.snapshot_path))

    grids = [torch.tensor(grid, requires_grad=False).cuda() for grid in data.grids]

    if not osp.exists(opts.model_dir):
        os.makedirs(opts.model_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.model_dir))
    if not osp.exists(opts.runs_dir):
        os.makedirs(opts.runs_dir, exist_ok=True)
        LOG_INFO('"%s" directory created' % (opts.runs_dir))
    writer = SummaryWriter(log_dir=opts.runs_dir)

    total_iters = len(data)*start_epoch//opts.batch_size

    for epoch in range(start_epoch, epoch_total):
        # training
        net.train()
        train_loss = 0
        epoch_loss = 0
        LOG_INFO('\nEpoch: %d' % epoch)
        # acc_total=0
        for step, data_blob in enumerate(dbloader):
            start_time = time.time()
            imgs, gt, valid, raw_imgs = data_blob

            imgs = [img.cuda() for img in imgs]
            valid = valid.cuda()
            gt = gt.cuda()

            # net.zero_grad()
            optimizer.zero_grad()

            predictions = net(imgs, grids, opts.train_iters)

            loss = sequence_loss(predictions, gt.unsqueeze(1), valid.unsqueeze(1))

            train_loss += loss.data

            epoch_loss = train_loss / (step + 1)

            if step % 200 == 0:
                LOG_INFO("Iter %d training loss = %.3f, average training loss for every step = %.3f, \
                    time = %.2f" % (total_iters, loss, epoch_loss, time.time() - start_time))
                writer.add_scalar("train/loss", loss, total_iters)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            total_iters += 1

        # logging
        invdepth_idx = torch.clamp(predictions[-1][0][0], 0, opts.net_opts.num_invdepth - 1)
        writer.add_scalar("train/epoch_loss", epoch_loss, total_iters)
        invdepth = data.indexToInvdepth(toNumpy(invdepth_idx))
        raw_imgs = [toNumpy(raw[0]) for raw in raw_imgs]
        vis_img = data.makeVisImage(raw_imgs, invdepth, gt=toNumpy(gt[0]))
        writer.add_image("train/vis", vis_img.transpose(2, 0, 1), total_iters)

        # evaluation
        net.eval()
        eval_list = data.opts.test_idx
        errors = np.zeros((len(eval_list), 5))
        for d in range(len(eval_list)):
            fidx = eval_list[d]
            imgs, gt, valid, raw_imgs = data.loadSample(fidx)
            imgs = [torch.Tensor(img).unsqueeze(0).cuda() for img in imgs]
            with torch.no_grad():
                invdepth_idx = net(imgs, grids, opts.valid_iters, test_mode=True)
            invdepth_idx = toNumpy(invdepth_idx[0, 0])
            # Compute errors
            errors[d, :] = data.evalError(invdepth_idx, gt, valid)
            # Visualization
        # logging
        invdepth = data.indexToInvdepth(invdepth_idx)
        raw_imgs = [toNumpy(raw) for raw in raw_imgs]
        vis_img = data.makeVisImage(raw_imgs, invdepth, gt=toNumpy(gt))
        writer.add_image("val/vis", vis_img.transpose(2, 0, 1), total_iters)

        mean_errors = errors.mean(axis=0)
        writer.add_scalar("val/>1", mean_errors[0], total_iters)
        writer.add_scalar("val/>3", mean_errors[1], total_iters)
        writer.add_scalar("val/>5", mean_errors[2], total_iters)
        writer.add_scalar("val/MAE", mean_errors[3], total_iters)
        writer.add_scalar("val/RMS", mean_errors[4], total_iters)
        LOG_INFO('>1: %.3f, >3: %.3f, >5: %.3f, MAE: %.3f, RMS: %.3f' %
            (mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4]))

        # save
        savefilename = opts.model_dir + '/%s_e%d.pth' % (opts.name, epoch)
        torch.save({
                'net_state_dict': net.state_dict(),
                'net_opts': opts.net_opts,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'epoch_loss': epoch_loss,
            }, savefilename)


def main():
    load_state = opts.snapshot_path is not None or opts.pretrain_path is not None
    train(opts.total_epochs, load_state)


if __name__ == "__main__":
    main()
