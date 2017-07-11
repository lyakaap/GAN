# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import initializers
import chainer.functions as F
import chainer.links as L
from pixel_shuffler import PixelShuffler

class Generator(chainer.Chain):
    def __init__(self):
        initializer = initializers.HeNormal()
        super(Generator, self).__init__(
                # num of noise that becomes the seed of Generation is 100 
            l0z = L.Linear(100, 7*7*128, initialW = initializer),
            ps1 = PixelShuffler(2, 128, 256),
            ps2 = PixelShuffler(2, 64, 4),
            bn0 = L.BatchNormalization(7*7*128),
            bn1 = L.BatchNormalization(64),
        )
        
    def make_hidden(self, batchsize):
        return numpy.random.normal(0, 1, (batchsize, 100, 1, 1))\
            .astype(numpy.float32)
        
    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.l0z(z))), (z.data.shape[0], 128, 7, 7))
        h = F.relu(self.bn1(self.ps1(h)))
        x = F.sigmoid((self.ps2(h)))
        return x


class Discriminator(chainer.Chain):
    def __init__(self):
        initializer = initializers.HeNormal()
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(1, 64, 4, stride=2, pad=1, initialW = initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW = initializer),
            l2 = L.Linear(7*7*128, 1, initialW = initializer), # 1 output (num of output of mattyaDCGAN is 2)
            bn1 = L.BatchNormalization(128),
        )
        
    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        #h = F.leaky_relu(self.bn1(self.c1(h)))
        l = self.l2(h)
        return l

