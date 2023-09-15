import os
import numpy as np
import math
from .image_dataset import ImageDataset


class HierarchicalCurveDataset(ImageDataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        self.debug = cmd.get('debug', False)
        self.is_training = True
        self.image_paths = []
        self.gt_paths = []
        
        self.parts = kwargs.get('parts', 3)
        self.points = kwargs.get('points', 2)
        
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            with open(self.data_list[i], 'r') as fid:
                image_list = fid.readlines()
            if self.is_training:
                image_path=[os.path.join(self.data_dir[i],'train_images',timg.strip()) for timg in image_list]  
                if 'CTW1500' in self.data_dir[0]:
                    gt_path=[os.path.join(self.data_dir[i],'bezier_train_gts',timg.strip().split('.')[0]+'.txt') for timg in image_list]
                else:
                    gt_path=[os.path.join(self.data_dir[i],'bezier_train_gts',timg.strip()+'.txt') for timg in image_list]
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
                parts = line.strip().split(',####')
                label = parts[-1]
                if 'CTW1500' in self.data_dir[0]:
                    label = '1'
                parts = parts[0].strip().split(',') + [label]
#                 parts = line.strip().split(',')
#                 label = parts[-1]
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                poly, polys = self._decode_polys(np.array(list(map(float, line[:-1]))))
                item['poly'] = poly
                item['seg_polys'] = polys
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res
    
    def _decode_polys(self, bezier):
        step = self.points+1
        u = np.linspace(0, 1, step*self.parts+1)
        bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
        points = np.outer((1 - u) ** 3, bezier[:, 0]) \
            + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
            + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
            + np.outer(u ** 3, bezier[:, 3])
        if points[:, 1].mean()<points[:, 3].mean():
            top = points[:, :2]
            bot = points[:, 2:]
        else:
            bot = points[:, :2]
            top = points[:, 2:]
        points = np.concatenate((top, bot), axis=0)
        
        points = self.check_clockwise(points)

        polys = []
        u_ind = 0
        b_ind = points.shape[0]
        for i in range(self.parts):
            poly = np.concatenate((points[u_ind:u_ind+step+1],points[b_ind-step-1:b_ind]), axis=0)
            polys.append(poly.reshape((-1, 2)).tolist())
            u_ind = u_ind+step
            b_ind = b_ind-step
        return points.reshape((-1, 2)).tolist(), polys
    
    def check_clockwise(self, points):
        y = points[:, 1]
        idx = np.argmax(y)
        x1 = points[(idx - 1 + len(points)) % len(points)]
        x2 = points[idx]
        x3 = points[(idx + 1) % len(points)]
        x2_x1 = x2 - x1
        x3_x2 = x3 - x2
        judge_result = x2_x1[0] * x3_x2[1] - x2_x1[1] * x3_x2[0]
        if judge_result < 0:
            points = points[::-1]
            half = points.shape[0]//2
            points = np.concatenate((points[half:], points[:half]), axis=0)

        return points
    
    
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls+1
    if n == 4:
        return np.array([[0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [0, 0, 128]]).astype(np.uint8)
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return np.array(palette[3:]).reshape(-1,3).tolist()