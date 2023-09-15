import numpy as np
import scipy.io as scio
from tqdm import tqdm
from .image_dataset import ImageDataset


class HierarchicalSynthDataset(ImageDataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''

    def __init__(self, data_dir=None, data_list=None, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        
        self.is_training = True
        
        self.debug = cmd.get('debug', False)
        
        self.image_paths = []
        self.targets = []
        
        self.parts = kwargs.get('parts', 3)

        print("Initializing SynthDataset")
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            data = scio.loadmat(self.data_list[i])

            img_paths = list(data['imnames'][0])
            gts = list(data['wordBB'][0])
            texts = list(data['txt'][0])

            img_paths=[self.data_dir[i]+'/img/'+timg[0].strip() for timg in img_paths]
            self.image_paths += img_paths
            
            for gt, text in tqdm(zip(gts,texts)):
                bboxes, words = self.load_ann(gt, text)
                
                lines = []
                for k in range(bboxes.shape[0]):
                    polys = self._decode_polys(bboxes[k])
                    item = {
                        'seg_polys':polys,
                        'poly':bboxes[k].tolist(),
                        'text':words[k]
                    }
                    lines.append(item)
                self.targets.append(lines)
        self.num_samples = len(self.image_paths)

    def load_ann(self, gt, text):
        bboxes = np.array(gt)
        bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
        bboxes = bboxes.transpose(2, 1, 0)
        bboxes = np.round(bboxes)

        words = []
        for text_ in text:
            text_ = text_.replace('\n', ' ').replace('\r', ' ')
            words.extend([w for w in text_.split(' ') if len(w) > 0])

        return bboxes, words
    
    def _decode_polys(self, bbox):
        step = 1
#         bbox = np.array(bbox).reshape(-1)
#         assert bbox.shape[0]==8
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
