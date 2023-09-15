import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import backbones
import decoders


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, parallelized, distributed, local_rank, device):
    if parallelized:
        if distributed:
            model.to(device)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=[local_rank])
        else:
            return nn.DataParallel(model)
    else:
        return nn.DataParallel(model, device_ids=[0])

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, parallelized: bool = True, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        self.model = parallelize(self.model, parallelized, distributed, local_rank, device)
        
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        if not distributed:
            self.criterion = parallelize(self.criterion, parallelized, distributed, local_rank, device)
        
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True, global_iter=0, **kwards):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, training=self.training)

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch, global_iter=global_iter)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred