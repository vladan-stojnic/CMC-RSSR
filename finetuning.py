from __future__ import print_function

import json
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import wandb
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import distributed
from torchvision import transforms

from dataset import (ClassificationImageDataset,
                     MultilabelClassificationImageDataset,
                     MultispectralRandomHorizontalFlip,
                     MultispectralRandomResizedCrop, MultispectralResize,
                     RGB2Lab, ScalerPCA)
from models.alexnet import alexnet, multispectral_alexnet
from models.LinearModel import (LinearClassifierAlexNet,
                                LinearClassifierResNetV2)
from models.resnet import ResNetV2, multispectral_ResNetV2
from spawn import spawn
from util import AverageMeter, adjust_learning_rate, parse_option


def get_train_val_loader(args):
    if not args.multispectral:    
        multilabel_targets = None
        target_transform = None
        task_type = 'single-label'

        if args.multilabel_targets:
            with open(args.multilabel_targets, 'r') as f:
                multilabel_targets = json.load(f)
                target_transform = torch.tensor
                task_type = 'multi-label'

        normalize = transforms.Normalize(mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                                     std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2])

        train_dataset = ClassificationImageDataset(
            args.data_folder,
            args.train_image_list,
            transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.0)),
                transforms.RandomHorizontalFlip(),
                RGB2Lab(),
                transforms.ToTensor(),
                normalize,
            ]),
            target_transform=target_transform,
            multilabel_targets=multilabel_targets
        )
        val_dataset = ClassificationImageDataset(
            args.data_folder,
            args.val_image_list,
            transforms.Compose([
                transforms.Resize(224),
                RGB2Lab(),
                transforms.ToTensor(),
                normalize,
            ]),
            target_transform=target_transform,
            multilabel_targets=multilabel_targets
        )
    else:
        task_type = 'single-label'
        target_transform = None
        
        if args.multispectral_dataset == 'BigEarthNet':
            target_transform = torch.tensor
            task_type = 'multi-label'
            
        train_dataset = MultilabelClassificationImageDataset(
            args.data_folder,
            args.train_image_list,
            transforms.Compose([
                MultispectralResize((256, 256)),
                MultispectralRandomResizedCrop(224, scale=(args.crop_low, 1.0)),
                MultispectralRandomHorizontalFlip(),
                ScalerPCA('./scaler_pca', args.pca),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform,
            dataset=args.multispectral_dataset
        )
        val_dataset = MultilabelClassificationImageDataset(
            args.data_folder,
            args.val_image_list,
            transforms.Compose([
                MultispectralResize((224, 224)),
                ScalerPCA('./scaler_pca', args.pca),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform,
            dataset=args.multispectral_dataset
        )
    
    print('number of train: {}'.format(len(train_dataset)))
    print('number of val: {}'.format(len(val_dataset)))

    if args.distributed:
        train_sampler = distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, train_sampler, train_dataset.num_classes, task_type


def set_model(args, ngpus_per_node, num_classes, task_type):
    if args.model == 'alexnet':
        if args.multispectral:
            model = multispectral_alexnet(args.feat_dim)
        else:
            model = alexnet(args.feat_dim)
        classifier = LinearClassifierAlexNet(layer=args.layer, n_label=num_classes, pool_type='avg')
    elif args.model.startswith('resnet'):
        if args.multispectral:
            model = multispectral_ResNetV2(args.model)
        else:
            model = ResNetV2(args.model)
        classifier = LinearClassifierResNetV2(layer=args.layer, n_label=num_classes, pool_type='avg')
    else:
        raise NotImplementedError(args.model)

    # load pre-trained model
    if not args.resume:
        print('==> loading pre-trained model')
        ckpt = torch.load(args.model_path)
        state_dict = ckpt['model']

        has_module = False
        for k, v in state_dict.items():
            if k.startswith('module'):
                has_module = True

        if has_module:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        print('==> done')

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            classifier.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int(args.num_workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            classifier.cuda()
            classifier = torch.nn.parallel.DistributedDataParallel(classifier)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        classifier = torch.nn.DataParallel(classifier).cuda()

    if task_type == 'single-label':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif task_type == 'multi-label':
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

    return model, classifier, criterion


def set_optimizer(args, model, classifier):
    if args.opt == 'adam':
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()),
                              lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()),
                              lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, classifier, criterion, optimizer, task_type, opt):
    """
    one epoch training
    """
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    predictions = []
    ground_truth = []

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float()
        input = input.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        feat_l, feat_ab = model(input, opt.layer)
        feat = torch.cat((feat_l, feat_ab), dim=1)

        output = classifier(feat)
        loss = criterion(output, target)

        losses.update(loss.item(), input.size(0))
        
        if task_type == 'single-label':
            predictions.append(F.softmax(output).cpu().detach().numpy().argmax(axis=1))
        elif task_type == 'multi-label':
            predictions.append(F.sigmoid(output).cpu().detach().numpy())
        ground_truth.append(target.cpu().numpy())

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
            
    predictions = np.concatenate(predictions, axis = 0)
    ground_truth = np.concatenate(ground_truth, axis = 0)
    if task_type == 'single-label':
        metric = accuracy_score(ground_truth, predictions)
    elif task_type == 'multi-label':
        metric = average_precision_score(ground_truth, predictions, average='macro')
    
    print(metric)

    return metric, losses.avg


def validate(val_loader, model, classifier, criterion, task_type, opt):
    """
    evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    predictions = []
    ground_truth = []

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            feat_l, feat_ab = model(input, opt.layer)
            feat = torch.cat((feat_l, feat_ab), dim=1)
            output = classifier(feat)
            loss = criterion(output, target)

            # measure loss
            losses.update(loss.item(), input.size(0))
            
            if task_type == 'single-label':
                predictions.append(F.softmax(output).cpu().numpy().argmax(axis=1))
            elif task_type == 'multi-label':
                predictions.append(F.sigmoid(output).cpu().numpy())
            ground_truth.append(target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses))

    predictions = np.concatenate(predictions, axis = 0)
    ground_truth = np.concatenate(ground_truth, axis = 0)
    if task_type == 'single-label':
        metric = accuracy_score(ground_truth, predictions)
    elif task_type == 'multi-label':
        metric = average_precision_score(ground_truth, predictions, average='macro')
    
    print(metric)

    return metric, losses.avg


def main():
    args = parse_option()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # set the data loader
    train_loader, val_loader, train_sampler, num_classes, task_type = get_train_val_loader(args)

    # set the model
    model, classifier, criterion = set_model(args, ngpus_per_node, num_classes, task_type)

    # set optimizer
    optimizer = set_optimizer(args, model, classifier)

    cudnn.benchmark = True

    # optionally resume
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'"
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #wandb.init(config=args, project=args.wandb_project)

    #wandb.watch(model)
    #wandb.watch(classifier)
    
    if args.evaluate:
        test_metric, test_loss = validate(val_loader, model, classifier, criterion, task_type, args)
        return

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_metric, train_loss = train(epoch, train_loader, model, classifier, criterion, optimizer, task_type, args)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        #wandb.log({'train_loss': train_loss, 'train_metric': train_metric}, step = epoch)

        print("==> testing...")
        test_metric, test_loss = validate(val_loader, model, classifier, criterion, task_type, args)

        #wandb.log({'val_loss': test_loss, 'val_metric': test_metric}, step = epoch)

    # save the model
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        state = {'epoch': epoch,
                 'classifier': classifier.state_dict(),
                 'model' : model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        print('saving model!')
        torch.save(state, args.save_path)


if __name__ == '__main__':
    main()
