import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from concern.config import State
from .data_process import DataProcess


class MakeDistMaps(DataProcess):
    bg_value = State(default=640)
    
    def __init__(self, cmd={}, *args, **kwargs):
        self.load_all(cmd=cmd, **kwargs)
    
    def process(self, data):
        gt = data['gt']
        dist_gt = self.get_dist_map(gt)
        data.update(dist_gt=dist_gt)
        if 'seg_polygons' in data:
            seg_gts = data['seg_gts']
            dist_seg_gts = self.get_dist_map(seg_gts)
            data.update(dist_seg_gts=dist_seg_gts)
        return data
    
    def get_dist_map(self, mask):
        dis_map = np.zeros(mask.shape, dtype=np.float32)
        binary_mask = np.array(mask==1).astype(np.uint8)
        for i in range(mask.shape[0]):
            dist = distance_transform_edt(binary_mask[i])
#             dist[mask[i]==0] = self.bg_value
            dist[mask[i]==0] = dist.max()
            dist = np.clip(dist, 1, self.bg_value)
            dis_map[i] = dist
        return dis_map
