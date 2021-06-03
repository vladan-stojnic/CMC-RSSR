from __future__ import print_function

import os
import sys
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from dataset import (ImageDataset, MultispectralImageDataset,
                     MultispectralRandomHorizontalFlip,
                     MultispectralRandomResizedCrop, RGB2Lab, ScalerPCA)
from models.alexnet import alexnet, multispectral_alexnet
from models.resnet import ResNetV2, multispectral_ResNetV2
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCECriterion
from util import AverageMeter, adjust_learning_rate, parse_option

warnings.filterwarnings("ignore")


def get_train_loader(args):
    """get the train loader"""
    data_folder = args.data_folder
    image_list = args.image_list

    if not args.multispectral:
        normalize = transforms.Normalize(mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                                         std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2])

        transformations = [transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.)),
                           transforms.RandomHorizontalFlip()]

        if args.resize_image_aug:
            transformations.insert(0, transforms.Resize((256, 256)))

        transformations += [RGB2Lab(), transforms.ToTensor(), normalize]
        train_transform = transforms.Compose(transformations)
        train_dataset = ImageDataset(data_folder, image_list, transform=train_transform)
        train_sampler = None
    else:
        transformations = [MultispectralRandomResizedCrop(224, scale=(args.crop_low, 1.)),
                           MultispectralRandomHorizontalFlip()]
        
        transformations += [ScalerPCA('./scaler_pca', use_pca=args.pca), transforms.ToTensor()]
        train_transform = transforms.Compose(transformations)
        train_dataset = MultispectralImageDataset(data_folder, image_list, transform=train_transform)
        train_sampler = None

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # num of samples
    n_data = len(train_dataset)
    print('number of samples: {}'.format(n_data))

    return train_loader, n_data


def set_model(args, n_data):
    # set the model
    if args.model == 'alexnet':
        if args.multispectral:
            model = multispectral_alexnet(args.feat_dim)
        else:
            model = alexnet(args.feat_dim)
    elif args.model.startswith('resnet'):
        if args.multispectral:
            model = multispectral_ResNetV2(args.model)
        else:
            model = ResNetV2(args.model)
    else:
        raise ValueError('model not supported yet {}'.format(args.model))
    contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m)
    criterion_l = NCECriterion(n_data)
    criterion_ab = NCECriterion(n_data)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        contrast = contrast.cuda()
        criterion_ab = criterion_ab.cuda()
        criterion_l = criterion_l.cuda()
        cudnn.benchmark = True

    return model, contrast, criterion_ab, criterion_l


def set_optimizer(args, model):
    # return optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, contrast, criterion_l, criterion_ab, optimizer, opt):
    """
    one epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l_loss_meter = AverageMeter()
    ab_loss_meter = AverageMeter()
    l_prob_meter = AverageMeter()
    ab_prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)
        inputs = inputs.float()
        if torch.cuda.is_available():
            index = index.cuda(async=True)
            inputs = inputs.cuda()

        # ===================forward=====================
        feat_l, feat_ab = model(inputs)
        out_l, out_ab = contrast(feat_l, feat_ab, index)

        l_loss = criterion_l(out_l)
        ab_loss = criterion_ab(out_ab)
        l_prob = out_l[:, 0].mean()
        ab_prob = out_ab[:, 0].mean()

        loss = l_loss + ab_loss

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        losses.update(loss.item(), bsz)
        l_loss_meter.update(l_loss.item(), bsz)
        l_prob_meter.update(l_prob.item(), bsz)
        ab_loss_meter.update(ab_loss.item(), bsz)
        ab_prob_meter.update(ab_prob.item(), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'l_p {lprobs.val:.3f} ({lprobs.avg:.3f})\t'
                  'ab_p {abprobs.val:.3f} ({abprobs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, lprobs=l_prob_meter,
                   abprobs=ab_prob_meter))
            print(out_l.shape)
            sys.stdout.flush()

    return l_loss_meter.avg, l_prob_meter.avg, ab_loss_meter.avg, ab_prob_meter.avg


def main():

    # parse the args
    args = parse_option(True)

    # set the loader
    train_loader, n_data = get_train_loader(args)

    # set the model
    model, contrast, criterion_ab, criterion_l = set_model(args, n_data)

    # set the optimizer
    optimizer = set_optimizer(args, model)

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        l_loss, l_prob, ab_loss, ab_prob = train(epoch, train_loader, model, contrast, criterion_l, criterion_ab,
                                                 optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'contrast': contrast.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

        pass


if __name__ == '__main__':
    main()
