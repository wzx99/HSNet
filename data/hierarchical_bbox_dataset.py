import os
import numpy as np
import math
from .image_dataset import ImageDataset


class HierarchicalBboxDataset(ImageDataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''

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
        
        self.use_customer_dictionary = kwargs.get('customer_dictionary', None)
        if not self.use_customer_dictionary:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
        self.voc_size = len(self.CTLABELS)+1
        
        self.parts = kwargs.get('parts', 3)
        self.points = kwargs.get('points', 1)
        
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            with open(self.data_list[i], 'r') as fid:
                image_list = fid.readlines()
            if self.is_training:
                image_path=[os.path.join(self.data_dir[i],'train_images',timg.strip()) for timg in image_list]  
                if 'MLT17' in self.data_list[i]:
                    gt_path=[os.path.join(self.data_dir[i],'train_gts',timg.strip().split('.')[0]+'.txt') for timg in image_list]
                else:
                    gt_path=[os.path.join(self.data_dir[i],'train_gts',timg.strip()+'.txt') for timg in image_list]
            else:
                image_path=[os.path.join(self.data_dir[i],'test_images',timg.strip()) for timg in image_list]
                if 'TD500' in self.data_list[i]:
                    gt_path=[os.path.join(self.data_dir[i],'test_gts',timg.strip()+'.txt') for timg in image_list]
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
                parts = line.strip().split(',')
                label = parts[-1]
                if 'TD' in self.data_dir[0] and label == '1':
                    label = '###'
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                if 'ICDAR' in self.data_dir[0]:
                    poly = np.array(list(map(float, line[:8]))).reshape((-1, 2)).tolist()
                else:
                    num_points = math.floor((len(line) - 1) / 2) * 2
                    poly = np.array(list(map(float, line[:num_points]))).reshape((-1, 2)).tolist()
                item['poly'] = poly
                item['seg_polys'] = self._decode_polys(poly)
                item['text'] = label
                lines.append(item)
            res.append(lines)
        return res
    
    def _decode_polys(self, bbox):
        step = 1
        bbox = np.array(bbox).reshape(4,2)
        bbox = self.check_clockwise(bbox)
        bbox = self.check_bbox(bbox)
        bbox = np.array(bbox).reshape(-1)
        new_x_t = np.linspace(float(bbox[0]), float(bbox[2]), num=step*self.parts+1)
        new_y_t = np.linspace(float(bbox[1]), float(bbox[3]), num=step*self.parts+1)
        new_x_b = np.linspace(float(bbox[4]), float(bbox[6]), num=step*self.parts+1)
        new_y_b = np.linspace(float(bbox[5]), float(bbox[7]), num=step*self.parts+1)
        new_x = np.concatenate((new_x_t, new_x_b), axis=0)
        new_y = np.concatenate((new_y_t, new_y_b), axis=0)
        points = np.stack((new_x, new_y), axis=-1)

        polys = []
        u_ind = 0
        b_ind = points.shape[0]
        for i in range(self.parts):
            poly = np.concatenate((points[u_ind:u_ind+step+1],points[b_ind-step-1:b_ind]), axis=0)
            polys.append(poly.reshape((-1, 2)).tolist())
            u_ind = u_ind+step
            b_ind = b_ind-step
        return polys
    
    def check_bbox(self, bbox):
        xmin, xmax = min(bbox[:,0]), max(bbox[:,0])
        w = xmax-xmin
        ymin, ymax = min(bbox[:,1]), max(bbox[:,1])
        h = ymax-ymin
        d = np.sum((bbox-np.array([[xmin,ymin]]))**2,axis=1)
        start_ind = np.argmin(d)
        bbox = np.roll(bbox, -start_ind, axis=0)
        return bbox
    
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

        return points