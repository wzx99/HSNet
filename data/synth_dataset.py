import numpy as np
import scipy.io as scio
from .image_dataset import ImageDataset


class SynthDataset(ImageDataset):
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
        self.get_all_samples()

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            data = scio.loadmat(self.data_list[i])

            img_paths = list(data['imnames'][0])
            gts = list(data['wordBB'][0])
            texts = list(data['txt'][0])

            img_paths=[self.data_dir[i]+'/img/'+timg[0].strip() for timg in img_paths]
            self.image_paths += img_paths
            
            for gt, text in zip(gts,texts):
                bboxes, words = self.load_ann(gt, text)
                
                lines = []
                for k in range(bboxes.shape[0]):
                    item = {
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
