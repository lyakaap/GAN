# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))
        #z = Variable(xp.zeros((n_images, 100, 1), dtype=xp.float32))
        label = [i for i in range(rows) for j in range(cols)]
        
        with chainer.using_config('train', False):
            x = gen(z, label)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        # gen_output_activation_func is sigmoid
        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        # gen output_activation_func is tanh
        #x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 1, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W))
        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir +\
            '/image{:0>6}.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image