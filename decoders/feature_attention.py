import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from .transformer import TransformerEncoderLayer, TransformerEncoder
BatchNorm2d = nn.BatchNorm2d


def np_softmax(feature, axis=1, t=1):
    e_x = np.exp(feature/t)
    probs = e_x / (np.sum(e_x, axis=axis, keepdims=True)+1e-7)
    return probs

    
class Instance_Attention(nn.Module):
    def __init__(self, in_channel, inner_channel=128, sample_n=32, thresh=0.3, num_encoder_layers=3, normalize_before=False):
        super(Instance_Attention, self).__init__()
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.thresh = -np.log(1/thresh-1)
        self.sample_n = sample_n
        self.num_encoder_layers = num_encoder_layers
        
        self.torch_xy_embedding = {}
        self.embd_conv = nn.Sequential(
            nn.Conv2d(in_channel+2, in_channel, 3, padding=1),
            BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, self.inner_channel, 3, padding=1),)
        
        if self.num_encoder_layers>0:
            transformer_encoder_layer = TransformerEncoderLayer(self.inner_channel, nhead=8, dim_feedforward=512,
                                                    dropout=0.1, activation="relu", normalize_before=normalize_before)
            encoder_norm = nn.LayerNorm(self.inner_channel) if normalize_before else None
            self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_encoder_layers, encoder_norm)
            
        
        self.decoder = ObjectAttentionBlock(self.inner_channel, self.inner_channel, self.in_channel, 1)
        
    def forward(self, score, feature):
        pos_embd = self.pos_embd(feature)
        feature = torch.cat((feature,pos_embd),dim=1)
        feature = self.embd_conv(feature)
        
        key_feature, _, batch_mask, kernel_mask = self.sample_features(score, feature)
        
        if self.num_encoder_layers>0:
            key_feature_refine = self.transformer_encoder(key_feature, src_key_padding_mask=~batch_mask)
        else:
            key_feature_refine = key_feature
        
        refined_feature = self.decoder(feature, key_feature_refine.permute(1,2,0).unsqueeze(3), key_feature_refine.permute(1,2,0).unsqueeze(3), batch_mask.unsqueeze(1).unsqueeze(3))
#         refined_feature = self.decoder(feature, key_feature.permute(1,2,0).contiguous().unsqueeze(3), key_feature.permute(1,2,0).contiguous().unsqueeze(3), kernel_mask.unsqueeze(1).unsqueeze(3))
        
        return refined_feature
    
    def sample_features(self, score, feature):
        B, N, H, W = score.shape
        score = score.detach().cpu().numpy()
        bitmap = score>self.thresh
        bitmap = bitmap.reshape(B*N,H,W).astype(np.uint8)
        key_feature = torch.zeros((N*self.sample_n,B,self.inner_channel)).cuda()
        pos = torch.zeros((N*self.sample_n,B,2)).cuda()
        batch_mask = torch.zeros((B,N*self.sample_n)).bool().cuda()
        kernel_mask = torch.zeros((B,N*self.sample_n)).bool().cuda()
        for i in range(B*N):
            B_ = i//N
            N_ = i%N
            num_objects, labels, stats, centroids = cv2.connectedComponentsWithStats(bitmap[i], connectivity=8, ltype=None)
            centroids = torch.from_numpy(centroids/(np.array([[H,W]]))).cuda()
#             ind = np.argsort(stats[1:,-1])[::-1][:self.sample_n].tolist()
            ind = np.argsort(stats[:,-1])[::-1][:self.sample_n].tolist()
            for j in range(0,len(ind)):
                ind_j = ind[j]
                index = labels==ind_j
                if stats[ind_j,-1]==0:
                    break
                score_tmp = score[B_,N_] * index + (-1e7) * (1-index)
                score_tmp = np_softmax(score_tmp, axis=(0,1), t=np.sqrt(self.in_channel))
                score_tmp = torch.from_numpy(score_tmp).cuda().unsqueeze(0)
                feature_tmp = (feature[B_]*score_tmp).sum(dim=(1,2))
                key_feature[N_*self.sample_n + j,B_] = key_feature[N_*self.sample_n + j,B_] + feature_tmp
                pos[N_*self.sample_n + j,B_] = centroids[ind_j]
                batch_mask[B_, N_*self.sample_n + j] = True
                kernel_mask[B_, N_*self.sample_n + j] = N_==0
        return key_feature, pos, batch_mask, kernel_mask
    
    def pos_embd(self,feat_in):
        n, c, h, w = feat_in.shape
        e_key = str(n)+"_"+str(h)+"_"+str(w)
        if e_key not in self.torch_xy_embedding:
            x_range = torch.linspace(-1, 1, feat_in.shape[-1])
            y_range = torch.linspace(-1, 1, feat_in.shape[-2])
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([feat_in.shape[0], 1, -1, -1])
            x = x.expand([feat_in.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            self.torch_xy_embedding[e_key] = coord_feat
        return self.torch_xy_embedding[e_key].type_as(feat_in)



def ConvModule(inp, oup, ksize):
    return nn.Sequential(
        nn.Conv2d(inp, oup, ksize, 1, ksize//2, bias=False),
        BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

    def build_project(self, in_channels, channels, num_convs, use_conv_module):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats, value_feats, key_mask):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous() #B,HW,C

        key = self.key_project(key_feats)
        value = self.value_project(value_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1) #B,C,N
        key_mask = key_mask.reshape(*key_mask.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous() #B,HW,N
        sim_map = torch.matmul(query, key) #B,HW,N
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_mask = key_mask.expand(sim_map.shape)
        sim_map = sim_map * sim_mask + (-1e7) * torch.logical_not(sim_mask)
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context

    
class ObjectAttentionBlock(SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, out_channels, scale):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True)
        self.bottleneck = ConvModule(
            in_channels * 2,
            out_channels,
            1)

    def forward(self, query_feats, key_feats, value_feats, key_mask):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats, value_feats, key_mask)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output    


    
