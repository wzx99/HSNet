import torch

from concern.config import Configurable, State


class OptimizerScheduler(Configurable):
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.load('learning_rate', cmd=cmd, **kwargs)
        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']

    def create_optimizer(self, model):
        optimizer = getattr(torch.optim, self.optimizer)(
                model.parameters(), **self.optimizer_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer

    
class OptimizerScheduler2(Configurable):
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.load('learning_rate', cmd=cmd, **kwargs)
        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']

    def create_optimizer(self, model):
        
        cnn = []
        transformer = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if 'transformer' in name:
                transformer.append(param)
                print(f"{name} transformer")
            else:
                cnn.append(param)
        
        optimizer = getattr(torch.optim, self.optimizer)(
                [{'params': cnn}, {'params': transformer, 'lr': 0.00001}], **self.optimizer_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer
    