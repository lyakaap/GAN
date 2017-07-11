# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable


class DCGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        gen, dis = self.gen, self.dis

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        images = [batch[i][0] for i in range(batchsize)]
        label = [batch[i][1] for i in range(batchsize)]
        x_real = Variable(self.converter(images, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        
        y_real = dis(x_real, label)
        x_fake = gen(z, label)
        y_fake = dis(x_fake, label)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        gen_optimizer.update(self.loss_gen, gen, y_fake)
        
# sigmoid_cross_entropy(x,1) = softplus(-x)
# sigmoid_cross_entropy(x,0) = softplus(x)
