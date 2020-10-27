from torch import nn
from torch import optim
import torch
from utils.utils import get_config_attr
import numpy as np


class NaiveWrapper(nn.Module):
    def __init__(self, base, optimizer, cfg, **kwargs):
        super().__init__()
        self.net = base
        self.optimizer = optimizer
        self.cfg = cfg

        self.clip_grad = True
        self.spatial_feat_shape = (2048,7,7)
        self.bbox_feat_shape = (100,2048)
        self.bbox_shape = (100,4)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def forward_net(self, x, y, **kwargs):
        batch_size = x.size(0)
        bfeat_dim = np.prod(self.bbox_feat_shape)
        bbox_feats, bboxes = x[:, :bfeat_dim].view(batch_size, *self.bbox_feat_shape), \
                             x[:,bfeat_dim:].view(batch_size, *self.bbox_shape)
        captions, caption_lens, labels = y
        ret_dict = self.net(bbox_feats=bbox_feats, captions=captions, caption_lens=caption_lens, labels=labels,
                            bboxes=bboxes, **kwargs)

        return ret_dict

    def observe(self, x, y, optimize=True):
        self.optimizer.zero_grad()
        ret_dict = \
            self.forward_net(x, y)

        loss = ret_dict['loss']
        if optimize:
            loss.backward()
            self.optimizer.step()

        return ret_dict
