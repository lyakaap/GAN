# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import print_function
import argparse
import chainer.links as L 
import chainer.functions as F

from chainer import initializers

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda

import numpy

from MNIST_net import Discriminator

class DiscriminatorClassifier(chainer.Chain):
    def __init__(self):
        initializer = initializers.HeNormal()
        dis = Discriminator()
        chainer.serializers.load_npz('result/dis_iter_500000.npz', dis)
        super(DiscriminatorClassifier, self).__init__(
            c0 = L.Convolution2D(1, 64, 4, stride=2, pad=1, initialW=dis.c0.W.data, initial_bias=dis.c0.b.data),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=dis.c1.W.data, initial_bias=dis.c1.b.data),
            l2 = L.Linear(7*7*128, 10, initialW = initializer),
            bn1 = L.BatchNormalization(128),
        )
        self.c0.disable_update()
        
#    def __init__(self):
#        initializer = initializers.HeNormal()
#        super(DiscriminatorClassifier, self).__init__(
#            c0 = L.Convolution2D(1, 64, 4, stride=2, pad=1, initialW=initializer),
#            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
#            l2 = L.Linear(7*7*128, 10, initialW = initializer),
#            bn1 = L.BatchNormalization(128),
#        )
#        
    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        l = self.l2(h)
        return l

class DelGradient(object):
    name = 'DelGradient'
    def __init__(self, delTgt):
        self.delTgt = delTgt

    def __call__(self, opt):
        for name,param in opt.target.namedparams():
            for d in self.delTgt:
                if d in name:
                    grad = param.grad
                    with cuda.get_device(grad):
                        grad*=0

def main():
    parser = argparse.ArgumentParser(description='Chainer example: DCGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='classify_result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=50000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    DD = DiscriminatorClassifier()
    dis = L.Classifier(DD)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        # Copy the model to the GPU
        dis.to_gpu()

     # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(dis)
    # freeze dis weight
    optimizer.add_hook(DelGradient(["c0","c1"]))

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(ndim=3, scale=255.) # ndim=3 : (ch,width,height)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
     
     # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, dis, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    print(DD.c0.W)
    print(DD.c0.b)
    # Run the training
    trainer.run()
    print(DD.c0.W)
    print(DD.c0.b)

if __name__ == '__main__':
    main()