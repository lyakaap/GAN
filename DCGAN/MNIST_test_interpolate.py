# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

import chainer
from chainer import Variable

from MNIST_net import Generator

import os

import numpy as np
from PIL import Image

def main():
    
    # Set up a neural network to train
    gen = Generator()
    chainer.serializers.load_npz('result/gen_iter_500000.npz', gen)

    np.random.seed(0)
    xp = gen.xp
    z = np.random.normal(0.0,1.0,(100,100))
    for i in range(0,10):
        for j in range(1,10):
            # interpolate gradually
            #z[i*10 + j] = z[i*10] * 0.1 * (10-j)
            sub = z[10] - z[30]
            z[i*10 + j] = z[i*10] - 0.1 * j * sub
    z = Variable(xp.asarray(z.reshape(100,100,1,1), dtype = np.float32))
    
    with chainer.using_config('train', False):
        x = gen(z)
    x = chainer.cuda.to_cpu(x.data)

    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((10, 10, 1, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((10 * H, 10 * W))
    preview_dir = 'interpolate/preview'
    preview_path = preview_dir + '/phenomenon.png'
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    Image.fromarray(x).save(preview_path)

if __name__ == '__main__':
    main()