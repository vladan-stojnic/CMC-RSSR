from __future__ import print_function

import argparse
import os

import numpy as np
import torch


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def parse_option(CMC_training=False):

    parser = argparse.ArgumentParser('Command Line Arguments')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')

    # optimization
    parser.add_argument('--opt', type=str, default='adam', help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='250,300,350', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # resume path
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet', 'resnet50', 'resnet101'])
    parser.add_argument('--nce_k', type=int, default=4096)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--layer', type=int, default=5, help='which layer to evaluate')

    # specify folder
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--model_path', type=str, default=None, help='path to save model')

    # specify list of images to use
    parser.add_argument('--image_list', type=str, default=None, help='list of images to use')
    parser.add_argument('--train_image_list', type=str, default=None, help='list of images for training of the finetuned model')
    parser.add_argument('--val_image_list', type=str, default=None, help='list of images for validation of the finetuned model')

    # specify multilabel classification mapping
    parser.add_argument('--multilabel_targets', type=str, default=None, help='path to pickeled dictionary of image:target mappings for multilabel datasets')

    # specify path to save features
    parser.add_argument('--features_path', type=str, default=None, help='path to features')

    # specify wandb project name
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name')

    # specify path to features for training and validation/test datasets for linear classifier
    parser.add_argument('--train_data_path', type=str, default=None, help='path to train features')
    parser.add_argument('--val_data_path', type=str, default=None, help='path to val features')

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.08, help='low area in crop')

    parser.add_argument('--resize_image_aug', dest='resize_image_aug', action='store_true')
    parser.set_defaults(resize_image_aug=False)

    parser.add_argument('--evaluate', dest='evaluate', action='store_true')
    parser.set_defaults(evaluate=False)

    parser.add_argument('--multispectral', dest='multispectral', action='store_true')
    parser.set_defaults(multispectral=False)

    parser.add_argument('--pca', dest='pca', action='store_true')
    parser.set_defaults(pca=False)

    parser.add_argument('--multispectral_dataset', type=str, default='BigEarthNet', help='multispectral dataset to use')
    
    parser.add_argument('--save_path', type=str, default=None, help='path to save finetuned model')

    # parallel setting
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if CMC_training:
        opt.model_name = 'memory_nce_{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.nce_k, opt.model, opt.learning_rate,
                                                                        opt.weight_decay, opt.batch_size)

        opt.model_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.model_folder):
            os.makedirs(opt.model_folder)

    return opt


if __name__ == '__main__':
    meter = AverageMeter()
