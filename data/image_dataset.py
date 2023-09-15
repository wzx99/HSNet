import os
import functools
import logging
import bisect

import torch.utils.data as data
import cv2
from PIL import Image 
import numpy as np
import glob
from concern.config import Configurable, State
import math

# class _Meta(type(data.Dataset), type(Configurable)):
#     pass

# class ImageDataset(data.Dataset, Configurable, metaclass=_Meta):
class ImageDataset(data.Dataset, Configurable):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''
    data_dir = State()
    data_list = State()
    processes = State(default=[])

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            with open(self.data_list[i], 'r') as fid:
                image_list = fid.readlines()
            if self.is_training:
                image_path = [os.path.join(self.data_dir[i], 'train_images', timg.strip()) for timg in image_list]
                if 'ICDAR2013' in self.data_list[i]:
                    gt_path = [os.path.join(self.data_dir[i], 'train_gts', 'gt_' + timg.strip().split('.')[0] + '.txt') for timg in image_list]
                elif 'EnsText' in self.data_list[i]:
                    gt_path = [os.path.join(self.data_dir[i], 'train_gts', timg.strip().split('.')[0] + '.txt') for timg in image_list]
                else:
                    gt_path = [os.path.join(self.data_dir[i],'train_gts',timg.strip()+'.txt') for timg in image_list]
            else:
                image_path=[os.path.join(self.data_dir[i],'test_images',timg.strip()) for timg in image_list]
                if 'TD500' in self.data_list[i] or 'total_text' in self.data_list[i]:
                    gt_path = [os.path.join(self.data_dir[i],'test_gts',timg.strip()+'.txt') for timg in image_list]
                elif 'CTW1500' in self.data_list[i]:
#                     gt_path=[os.path.join(self.data_dir[i],'test_gts','000'+timg.strip().split('.')[0]+'.txt') for timg in image_list]
                    gt_path = [os.path.join(self.data_dir[i], 'test_gts', timg.strip().split('.')[0]+'.txt') for timg in image_list]
                elif 'EnsText' in self.data_list[i]:
                    gt_path = [os.path.join(self.data_dir[i], 'test_gts', timg.strip().split('.')[0] + '.txt') for timg in image_list]
                else:
                    gt_path=[os.path.join(self.data_dir[i],'test_gts','gt_'+timg.strip().split('.')[0]+'.txt') for timg in image_list]
            self.image_paths += image_path
            self.gt_paths += gt_path
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r', encoding='utf8').readlines()
            for line in reader:
                item = {}
                if 'CTW1500' in self.data_dir[0]:
#                     parts = line.strip().split(',####')
#                     label = parts[-1]
#                     label = '####' + label
#                     parts = parts[0].strip().split(',') + [label]
                    label = '1'
                    parts = line.strip().split(',') + [label]
                elif 'EnsText' in self.data_dir[0]:
                    label = '1'
                    parts = line.strip().split(',') + [label]
                else:
                    parts = line.strip().split(',')
                    label = parts[-1]
                if 'TD' in self.data_dir[0] and label == '1':
                    label = '###'
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                if 'ICDAR2015' in self.data_dir[0]:
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                elif 'ICDAR2013' in self.data_dir[0]:
                    x1, y1, x2, y2 = map(float, line[:4])
                    poly = np.array([x1, y1, x1, y2, x2, y2, x2, y1]).reshape((-1, 2)).tolist()
                else:
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            img = Image.open(image_path).convert('RGB')
        else:
            img = img[:,:,::-1]
            # pass  # for dbnet official model
        img = np.array(img).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target
        if self.processes is not None:
            for data_process in self.processes:
                data = data_process(data)
        return data

    def __len__(self):
        return len(self.image_paths)
