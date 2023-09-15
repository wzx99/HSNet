from concern.config import Configurable, State
import os
import torch


class Checkpoint(Configurable):
    start_epoch = State(default=0)
    start_iter = State(default=0)
    resume = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        cmd = kwargs['cmd']
        if 'start_epoch' in cmd:
            self.start_epoch = cmd['start_epoch']
        if 'start_iter' in cmd:
            self.start_iter = cmd['start_iter']
        if 'resume' in cmd:
            self.resume = cmd['resume']

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            logger.warning("Checkpoint not found: " +
                                self.resume)
            return

        logger.info("Resuming from " + self.resume)
        state_dict = torch.load(self.resume, map_location=device)
        
        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys

        missing_keys_list = []
        for k in missing_keys:
            if not k.endswith('.num_batches_tracked'):
                missing_keys_list.append(k)
        if len(missing_keys_list) > 0:
            logger.warning('Missing key(s) in state_dict: {}'.format(
                ', '.join('{}'.format(k) for k in missing_keys_list)))

        unexpected_keys_list = []
        for k in unexpected_keys:
            if not k.endswith('.num_batches_tracked'):
                unexpected_keys_list.append(k)
        if len(unexpected_keys_list) > 0:
            logger.warning('Unexpected key(s) in state_dict: {}'.format(
                ', '.join('{}'.format(k) for k in unexpected_keys_list)))
            
        model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed from " + self.resume)

    def restore_counter(self):
        return self.start_epoch, self.start_iter
