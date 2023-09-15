from collections import OrderedDict

import torch
import torch.nn as nn
from .feature_attention import Instance_Attention
BatchNorm2d = nn.BatchNorm2d


class HierarchicalDetector(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, parts=2,
                 bias=False, attention_type='scale_spatial',
                 sample_n=32, thresh=0.4, num_encoder_layers=3,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(HierarchicalDetector, self).__init__()
        self.parts = parts
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels//4, 3, padding=1, bias=bias)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

        self.concat_attention = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, bias=bias, padding=1),
            BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True))
        
        self.coarse_binarize_ablation = nn.Sequential(
            nn.Conv2d(inner_channels // 4, inner_channels // 4, 3, bias=bias, padding=1),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels//4, 1+self.parts, 1),)
#             nn.Sigmoid())
        
        self.instance_attention = Instance_Attention(inner_channels // 4, sample_n=sample_n, thresh=thresh, num_encoder_layers=num_encoder_layers, normalize_before=False)

        self.final_binarize_ablation = nn.Sequential(
            nn.Conv2d(inner_channels // 4, inner_channels // 4, 3, bias=bias, padding=1),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1+self.parts, 2, 2),
            nn.Sigmoid())
        
        self.coarse_binarize_ablation.apply(self.weights_init)
        self.final_binarize_ablation.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, features, gt=None, masks=None, training=False):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        fuse = self.concat_attention(fuse)
        
        coarse_score = self.coarse_binarize_ablation(fuse)
        
        final_fuse = self.instance_attention(coarse_score, fuse)

        final_score = self.final_binarize_ablation(final_fuse)
        
        if self.training:
            coarse_score = torch.sigmoid(coarse_score)
            result = OrderedDict(binary=final_score, deepsup=coarse_score)
        else:
#             final_score = nn.functional.interpolate(coarse_score, size=final_score.size()[2:], mode='bilinear', align_corners=False)
#             final_score = torch.sigmoid(final_score)
            return final_score
        return result
