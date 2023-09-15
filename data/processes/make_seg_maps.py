import numpy as np
import cv2
import os

from concern.config import State
from .data_process import DataProcess


class MakeSegMaps(DataProcess):
    parts = State(default=2)
    
    def __init__(self, cmd={}, *args, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
    
    def process(self, data):
        image = data['image']
        polygons = data['seg_polygons']
        ignore_tags = data['ignore_tags']
        
        h, w = image.shape[:2]
        seg_gts = np.zeros((self.parts, h, w), dtype=np.float32)
        for i in range(len(polygons)):
            if ignore_tags[i]:
                continue
            polygon = polygons[i]
            for part in range(self.parts):
                poly_part = polygon[part]
                poly_part = np.array(poly_part).reshape(-1, 2)
                cv2.fillPoly(seg_gts[part], [poly_part.astype(np.int32)], 1)
            
        data.update(seg_gts=seg_gts)
        return data
    
    
class MakeInstanceSegMaps(DataProcess): 
    sample_n = State(default=32)
    
    def process(self, data):
        image = data['image']
        gt = data['gt'].astype(np.uint8)
        
        h, w = image.shape[:2]
        instance_gts = np.zeros((self.sample_n, h, w), dtype=np.float32)
        valid_instance_num = 0
        num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(gt[0], connectivity=8, ltype=None)
        ind = np.argsort(stats[:,-1])[::-1][:self.sample_n].tolist()
        for j in range(0,len(ind)):
            ind_j = ind[j]
            if stats[ind_j,-1]==0:
                break
            instance_gts[j] = (labels==ind_j).astype(np.float32)
            valid_instance_num += 1
        data.update(instance_gts=instance_gts, valid_instance_num=valid_instance_num)
        return data