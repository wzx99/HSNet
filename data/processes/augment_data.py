import imgaug
import numpy as np
import random

from concern.config import State
from .data_process import DataProcess
from data.augmenter import AugmenterBuilder
import cv2
import math
import os ###

class AugmentData(DataProcess):
    augmenter_args = State(autoload=False)

    def __init__(self, **kwargs):
        self.augmenter_args = kwargs.get('augmenter_args')
        self.keep_ratio = kwargs.get('keep_ratio')
        self.only_resize = kwargs.get('only_resize')
        self.augmenter = AugmenterBuilder().build(self.augmenter_args)

    def may_augment_annotation(self, aug, data):
        pass

    def resize_image(self, image):
        origin_height, origin_width, _ = image.shape
        resize_shape = self.augmenter_args[0][1]
        height = resize_shape['height']
        width = resize_shape['width']
        if self.keep_ratio:
            width = origin_width * height / origin_height
            N = math.ceil(width / 32)
            width = N * 32
        image = cv2.resize(image, (width, height))
        return image

    def process(self, data):
        image = data['image']
        aug = None
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if self.only_resize:
                data['image'] = self.resize_image(image)
            else:
                data['image'] = aug.augment_image(image)
            self.may_augment_annotation(aug, data, shape)

        filename = data.get('filename', data.get('data_id', ''))
        data.update(filename=filename, shape=shape[:2])
        if not self.only_resize:
            data['is_training'] = True 
        else:
            data['is_training'] = False 
        return data


class AugmentDetectionData(AugmentData):
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',
                'text': line['text'],
            })
        data['polys'] = line_polys
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly
    
    
class AugmentDetectionSegData(AugmentDetectionData):
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for line in data['lines']:
            if self.only_resize:
                new_poly = [(p[0], p[1]) for p in line['poly']]
                seg_new_poly = []
                for poly in line['seg_polys']:
                    seg_new_poly.append([(p[0], p[1]) for p in poly])
            else:
                new_poly = self.may_augment_poly(aug, shape, line['poly'])
                seg_new_poly = []
                for poly in line['seg_polys']:
                    seg_new_poly.append(self.may_augment_poly(aug, shape, poly))
            line_polys.append({
                'points': new_poly,
                'ignore': line['text'] == '###',
                'text': line['text'],
                'seg_points': seg_new_poly,
            })
        data['polys'] = line_polys
        return data

    
class AugmentFlipSegData(DataProcess):
    flip_p = State(autoload=False)

    def __init__(self, **kwargs):
        self.flip_p = kwargs.get('flip_p', 0)

    def process(self, data):
        if random.random() < self.flip_p:
            image = data['image']
            data['image'] = image[:,::-1]
            
            width = image.shape[1]
            lines = []
            for line in data['lines']:
                new_poly = [(width-p[0], p[1]) for p in line['poly']]
                seg_new_poly = []
                for poly in line['seg_polys']:
                    seg_new_poly.append([(width-p[0], p[1]) for p in poly])
                seg_new_poly = seg_new_poly[::-1]
                line.update({'poly': new_poly, 'seg_polys': seg_new_poly})
                lines.append(line)
            data['lines'] = lines
        return data
            
#         import os
#         img = data['image'].astype(np.uint8).copy()
#         for line in data['lines']:
#             polys = line['seg_polys']
#             for scatter in np.array(polys[0]).astype(int).tolist():
#                 cv2.circle(img,scatter, 5, (0,0, 255), -1)
#             for scatter in np.array(polys[1]).astype(int).tolist():
#                 cv2.circle(img,scatter, 5, (255,0, 0), -1)
#         save_path = '/gdata/wangzx/tmp/img/' + os.path.split(data['filename'])[-1]
#         cv2.imwrite(save_path, img)
#         return data
