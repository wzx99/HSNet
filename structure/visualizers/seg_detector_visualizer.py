import cv2
import os
# import concern.webcv2 as webcv2
import numpy as np
import torch

from concern.config import Configurable, State
from data.processes.make_icdar_data import MakeICDARData
from data.processes.normalize_image import NormalizeImage


class SegDetectorVisualizer(Configurable):
    vis_num = State(default=4)
    eager_show = State(default=False)

    def __init__(self, **kwargs):
        cmd = kwargs['cmd']
        if 'eager_show' in cmd:
            self.eager_show = cmd['eager_show']

    def visualize(self, batch, output_pair, pred):
        boxes, _ = output_pair
        result_dict = {}
        for i in range(batch['image'].size(0)):
            result_dict.update(
                self.single_visualize(batch, i, boxes[i], pred))
        if self.eager_show:
#             webcv2.waitKey()
            return {}
        return result_dict

    def _visualize_heatmap(self, heatmap, canvas=None):
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap[0] * 255).astype(np.uint8)
        if canvas is None:
            pred_image = heatmap
        else:
            pred_image = (heatmap.reshape(
                *heatmap.shape[:2], 1).astype(np.float32) / 255 + 1) / 2 * canvas
            pred_image = pred_image.astype(np.uint8)
        return pred_image
    
    def _visualize_part_heatmap(self, heatmap, canvas=None):
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
            
        n = heatmap.shape[0]
        if n==4:
            palette = np.array([[  0,   0,   0],
                           [128,   0,   0],
                           [  0, 128,   0],
                           [0, 0,   128]]).astype(np.uint8)
        else:
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
            palette = np.array(palette).reshape(-1,3).astype(np.uint8)
        
        heatmap_tmp = heatmap>0.3
        heatmap = np.zeros((*heatmap.shape[1:],3),dtype=np.uint8)
        for i in range(1,n):
            heatmap = heatmap + palette[heatmap_tmp[i]*i]
        heatmap = np.clip(heatmap,a_min=0,a_max=255)
        if canvas is None:
            pred_image = heatmap
        else:
            pred_image = (heatmap.astype(np.float32) / 255 + 1) / 2 * canvas
            pred_image = pred_image.astype(np.uint8)
        return pred_image

    def single_visualize(self, batch, index, boxes, pred):
        image = batch['image'][index]
#         polygons = batch['polygons'][index]
#         if isinstance(polygons, torch.Tensor):
#             polygons = polygons.cpu().data.numpy()
        ignore_tags = batch['ignore_tags'][index]
        original_shape = batch['shape'][index]
        filename = os.path.basename(batch['filename'][index])
#         std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
#         mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
#         image = (image.cpu().numpy() * std + mean).transpose(1, 2, 0) * 255
        image = NormalizeImage.restore(image)
        image = image[:,:,::-1]
        pred_canvas = image.copy().astype(np.uint8)
        pred_canvas = cv2.resize(pred_canvas, (int(original_shape[1]), int(original_shape[0])))

#         if isinstance(pred, dict) and 'thresh' in pred:
#             thresh = self._visualize_heatmap(pred['thresh'][index])

#         if isinstance(pred, dict) and 'thresh_binary' in pred:
#             thresh_binary = self._visualize_heatmap(pred['thresh_binary'][index])
#             MakeICDARData.polylines(self, thresh_binary, polygons, ignore_tags)
            
        if isinstance(pred, dict) and 'binary' in pred:
            pred_map = pred['binary'][index] 
        else:
            pred_map = pred[index] 
        pred_binary = self._visualize_heatmap(pred_map)
        
        if pred_map.shape[0]>1:
            pred_part = self._visualize_part_heatmap(pred_map)
        

        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 4)
#             if isinstance(pred, dict) and 'thresh_binary' in pred:
#                 cv2.polylines(thresh_binary, [box], True, (0, 255, 0), 1) 

        if self.eager_show:
            return {}
        else:
            return_dict = {
                filename + '_output': pred_canvas,
                filename + '_pred': pred_binary    
            }
            if isinstance(pred, dict) and 'thresh' in pred:
                return_dict[filename + '_thresh'] = thresh
                return_dict[filename + '_thresh_binary'] = thresh_binary
            if pred_map.shape[0]>1:
                return_dict[filename + '_part'] = pred_part
            return return_dict

    def demo_visualize(self, image_path, output):
        boxes, _ = output
        boxes = boxes[0]
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        original_shape = original_image.shape
        pred_canvas = original_image.copy().astype(np.uint8)
        pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)

        return pred_canvas

