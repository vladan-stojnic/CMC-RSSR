from __future__ import print_function

import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms

from dataset import (ClassificationImageDataset,
                     MultilabelClassificationImageDataset, MultispectralResize,
                     RGB2Lab, ScalerPCA)
from models.alexnet import alexnet, multispectral_alexnet
from models.resnet import ResNetV2, multispectral_ResNetV2
from util import parse_option


def get_loader(args):
    folder = args.data_folder
    image_list = args.image_list

    if not args.multispectral:
    
        multilabel_targets = None
        target_transform = None

        if args.multilabel_targets:
            with open(args.multilabel_targets, 'r') as f:
                multilabel_targets = json.load(f)
                target_transform = torch.tensor

        normalize = transforms.Normalize(mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                                         std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2])

        dataset = ClassificationImageDataset(
            folder,
            image_list,
            transforms.Compose([
                transforms.Resize((224, 224)),
                RGB2Lab(),
                transforms.ToTensor(),
                normalize
            ]),
            target_transform=target_transform,
            multilabel_targets=multilabel_targets
        )
    else:

        target_transform = None
    
        if args.multispectral_dataset == 'BigEarthNet':
            target_transform = torch.tensor
            
        dataset = MultilabelClassificationImageDataset(
            folder,
            image_list,
            transforms.Compose([
                MultispectralResize((224, 224)),
                ScalerPCA('./scaler_pca', args.pca),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform,
            dataset=args.multispectral_dataset
        )
    
    print('number of images: {}'.format(len(dataset)))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return loader


def set_model(args):
    if args.model == 'alexnet':
        if args.multispectral:
            model = multispectral_alexnet(args.feat_dim)
        else:
            model = alexnet(args.feat_dim)
    elif args.model == 'resnet50':
        if args.multispectral:
            model = multispectral_ResNetV2(args.model)
        else:
            model = ResNetV2(args.model)
    else:
        raise NotImplementedError(args.model)

    print('==> loading pre-trained model')
    ckpt = torch.load(args.resume)
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

    if args.model == 'resnet50':
        model = nn.DataParallel(model)
    print('==> done')
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    return model


def validate(val_loader, model, opt):
    """
    evaluation
    """
    features = []
    targets = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):
            print(idx)
            input = input.float()
            targets.append(target.numpy())
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()


            feat_l, feat_ab = model(input, opt.layer)
            feat = torch.cat((feat_l, feat_ab), dim=1)

            ff = f.adaptive_avg_pool2d(feat, (1, 1))
            features.append(ff.view(ff.size(0), -1).cpu().numpy())

    return features, targets

	
def main():
    # parsing args
    args = parse_option()

    # set the model
    model = set_model(args)

    # set the data loader
    loader = get_loader(args)
	
    features, targets = validate(loader, model, args)
    features = np.concatenate(features, axis = 0)
    targets = np.concatenate(targets, axis = 0)
    print(features.shape)
    print(targets.shape)
	
    np.savez(args.features_path, features, targets)

if __name__ == '__main__':
    main()
