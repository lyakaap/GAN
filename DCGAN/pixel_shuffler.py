# -*- coding: utf-8 -*-
#!/usr/bin/env python

from chainer import functions as F
from chainer import links as L
from chainer import link
import chainer

class PixelShuffler(link.Link):
    def __init__(self, r, in_c, out_c):
        super(PixelShuffler, self).__init__()
        self.r = r
        self.conv = L.Convolution2D(in_c, out_c, 3, 1, 1)
        self.conv.W.data = chainer.cuda.to_gpu(self.conv.W.data)
        self.conv.b.data = chainer.cuda.to_gpu(self.conv.b.data)
        
    def __call__(self, x):
        r = self.r
        out = self.conv(x)
        batchsize = out.shape[0]
        in_channels = out.shape[1]
        out_channels = int(in_channels / (r ** 2))
        in_height = out.shape[2]
        in_width = out.shape[3]
        out_height = in_height * r
        out_width = in_width * r
        out = F.reshape(out, (batchsize, r, r, out_channels, in_height, in_width))
        out = F.transpose(out, (0, 3, 4, 1, 5, 2))
        out = F.reshape(out, (batchsize, out_channels, out_height, out_width))
        return out