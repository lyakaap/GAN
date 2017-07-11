# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
from PIL import Image
import chainer
import chainer.cuda
from chainer import Variable
import argparse
from MNIST_net import Generator


parser = argparse.ArgumentParser(description='DigitGenerator')
parser.add_argument('--digits', '-d', default='123',
                    help='Input some digits you want to generate')
args = parser.parse_args()

digits = args.digits

gen = Generator()
chainer.serializers.load_npz('result/gen_iter_100000.npz', gen)

xp = gen.xp
z = Variable(xp.asarray(gen.make_hidden(len(digits))))
label = [int(digits[i]) for i in range(len(digits))]
     
with chainer.using_config('train', False):
    x = gen(z, label)
x = chainer.cuda.to_cpu(x.data)

x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
# gen output_activation_func is tanh
#x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)
_, ch, H, W = x.shape
x = x.transpose(2, 0, 3, 1)
x = x.reshape((H, len(digits) * W))

Image.fromarray(x).show()