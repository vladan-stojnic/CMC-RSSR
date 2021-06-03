import math
import os
import random

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
import torch.utils.data as datautils
from joblib import load
from PIL import Image
from skimage import color


def loads_pyarrow(buff):
    return pa.deserialize(buff)


def my_resize_4(img, shape):
    out = np.zeros((shape[0], shape[1], 4))
    out[:, :, 0:3] = cv2.resize(img[:, :, 0:3], shape)
    out[:, :, 3] = cv2.resize(img[:, :, 3], shape)

    return out


def my_resize_6(img, shape):
    out = np.zeros((shape[0], shape[1], 6))
    out[:, :, 0:3] = cv2.resize(img[:, :, 0:3], shape)
    out[:, :, 3:6] = cv2.resize(img[:, :, 3:6], shape)

    return out


def my_resize_10(img, shape):
    out = np.zeros((shape[0], shape[1], 10))
    out[:, :, 0:3] = cv2.resize(img[:, :, 0:3], shape)
    out[:, :, 3:6] = cv2.resize(img[:, :, 3:6], shape)
    out[:, :, 6:9] = cv2.resize(img[:, :, 6:9], shape)
    out[:, :, 9] = cv2.resize(img[:, :, 9], shape)

    return out


def crop(img, top, left, height, width):
    return img[top:top+height, left:left+width, :]


def resized_crop(img, top, left, height, width, size):
    img = crop(img, top, left, height, width)
    img = my_resize_10(img, size)
    return img
    

class MultispectralResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return my_resize_10(img, self.size)


class MultispectralRandomHorizontalFlip(object):
    def __init__(self, p = 0.5):
        self. p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img[:, ::-1, :]
        return img


class ScalerPCA(object):
    def __init__(self, path, use_pca=True):
        self.scaler = load(path + '/' + 'scaler.pkl')
        self.pca = load(path + '/' + 'pca.pkl')
        self.use_pca = use_pca
    def __call__(self, img):
        img *= 30000
        img_shape = img.shape
        img = img.reshape((img_shape[0] * img_shape[1], img_shape[2]))
        img = self.scaler.transform(img)
        if self.use_pca:
            img = self.pca.transform(img)
        img = img.reshape((img_shape[0], img_shape[1], img_shape[2]))

        return img


class MultispectralRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = img.shape[0], img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)

                return i, j, h, w

        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        out = resized_crop(img, i, j, h, w, self.size)

        return out


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageDataset(datautils.Dataset):
    def __init__(self, root_path, images_to_use, transform=None):
        super(ImageDataset, self).__init__()

        with open(images_to_use, 'r') as f:
            images = f.readlines()

        self.samples = [os.path.join(root_path, image_path.strip()) for image_path in images]

        self.loader = pil_loader

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, 0, index


class MultispectralImageDataset(datautils.Dataset):
    def __init__(self, lmdb_path, images_to_use, transform=None):
        super(MultispectralImageDataset, self).__init__()

        with open(images_to_use, 'r') as f:
            names = f.readlines()

        self.names = [name.strip().split('.')[0] for name in names]

        self.env = lmdb.open(lmdb_path, readonly = True, lock = False, readahead = False, meminit = False)

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = np.zeros((256, 256, 10))

        with self.env.begin(write = False) as txn:
            byteflow = txn.get(self.names[idx].encode('ascii'))

        data_120, data_60 = loads_pyarrow(byteflow)
        
        data_120 = data_120.astype(np.float32).transpose((1, 2, 0))
        data_60 = data_60.astype(np.float32).transpose((1, 2, 0))
        
        data_120 = my_resize_4(data_120 / 30000, (256, 256))
        data_60 = my_resize_6(data_60 / 30000, (256, 256))

        sample[:, :, 0:3] = data_120[:, :, 0:3]
        sample[:, :, 3:6] = data_60[:, :, 0:3]
        sample[:, :, 6] = data_120[:, :, 3]
        sample[:, :, 7:] = data_60[:, :, 3:]

        if self.transform:
            sample = self.transform(sample)

        return sample, 0, idx


class FeatureClassificationDataset(datautils.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return self.features.shape[0]
        
    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def loader_big_earth_net(byteflow):
    sample = np.zeros((256, 256, 10))
    
    data_120, data_60, label = loads_pyarrow(byteflow)
        
    data_120 = data_120.astype(np.float32).transpose((1, 2, 0))
    data_60 = data_60.astype(np.float32).transpose((1, 2, 0))
        
    data_120 = my_resize_4(data_120 / 30000, (256, 256))
    data_60 = my_resize_6(data_60 / 30000, (256, 256))
    
    sample[:, :, 0:3] = data_120[:, :, 0:3]
    sample[:, :, 3:6] = data_60[:, :, 0:3]
    sample[:, :, 6] = data_120[:, :, 3]
    sample[:, :, 7:] = data_60[:, :, 3:]
    
    return sample, label


def loader_so2sat(byteflow):
    data, label = loads_pyarrow(byteflow)

    data = np.array(data)

    data /= 2.8
    data *= np.array([17561.21923828, 17054, 16508.984375, 16226.00726318, 
                      16108.16802979, 16047.890625, 15946.984375, 15849.51269531,
                      14715.36694336, 15145])
        
    data = my_resize_10(data / 30000, (256, 256))
    
    return data, label
    

class MultilabelClassificationImageDataset(datautils.Dataset):
    def __init__(self, lmdb_path, images_to_use, transform=None, target_transform=None, dataset='BigEarthNet'):
        super(MultilabelClassificationImageDataset, self).__init__()
        
        with open(images_to_use, 'r') as f:
            names = f.readlines()

        self.names = [os.path.split(name.strip())[1] for name in names]
        
        self.env = lmdb.open(lmdb_path, max_readers = 1, readonly = True, lock = False, readahead = False, meminit = False)
            
        self.transform = transform
        
        self.target_transform = target_transform
        
        if dataset == 'BigEarthNet':
            self.num_classes = 19
            self.loader = loader_big_earth_net
        elif dataset == 'So2Sat':
            self.num_classes = 17
            self.loader = loader_so2sat
        else:
            raise ValueError('Not supported dataset!!!')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        env = self.env

        with env.begin(write = False) as txn:
            byteflow = txn.get(self.names[idx].encode('ascii'))

        sample, target = self.loader(byteflow)
        
        if self.transform:
            sample = self.transform(sample)
            
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


class ClassificationImageDataset(datautils.Dataset):
    def __init__(self, root_path, images_to_use, transform=None, target_transform=None, multilabel_targets=None):
        super(ClassificationImageDataset, self).__init__()
        
        with open(images_to_use, 'r') as f:
            self.samples = f.readlines()
            
        self.samples.sort()
            
        self.samples = [os.path.join(root_path, image_path.strip()) for image_path in self.samples]
            
        self.loader = pil_loader

        self.transform = transform
        
        self.target_transform = target_transform
        
        if multilabel_targets:
            self.targets = self._make_targets(multilabel_targets=multilabel_targets)
        else:
            classes, class_to_idx = self._find_classes(root_path)
            self.targets = self._make_targets(class_to_idx=class_to_idx)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        path, target = self.samples[index], self.targets[index]

        img = self.loader(path)

        if self.transform:
            img = self.transform(img)
            
        if self.target_transform:
            target = self.target_transform(target)

        return img, target
            
    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
        
    def _make_targets(self, class_to_idx=None, multilabel_targets=None):
        if class_to_idx:
            self.num_classes = len(class_to_idx)
            return np.array([class_to_idx[os.path.split(os.path.split(sample)[0])[1]] for sample in self.samples])
            
        if multilabel_targets:
            self.num_classes = len(multilabel_targets[os.path.split(self.samples[0])[1]])
            return [multilabel_targets[os.path.split(sample)[1]] for sample in self.samples]
            
        raise ValueError("Either class_to_idx or multilabel_targets must be supplied!!!")
