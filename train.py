from mxnet.gluon import nn, utils as gutils
from mxnet import gluon, init, ndarray, autograd
import mxnet
import time
import os
import numpy
import random
import load
import model
import sys

batch_mode = True
if len(sys.argv) > 1:
    batch_mode = False

GPU_NUM = 2
def get_ctx():
    ctx = [mxnet.gpu(i) for i in range(GPU_NUM)]
    return ctx


def _get_batch(batch, ctx):
    """return features and labels on ctx"""

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    features,labels = batch
    labels = labels.astype('float16')
    return (gutils.split_and_load(features, ctx, even_split=False),

            gutils.split_and_load(labels, ctx, even_split=False),

            len(features))
    # return (features,labels,features.shape[0])


def evaluate_accuracy(data_iter, net, ctx=[mxnet.cpu()]):
    """Evaluate accuracy of a model on the given data set."""

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]

    acc = ndarray.array([0],dtype='float16')

    n = 0

    for batch in data_iter:

        features, labels, batch_size = _get_batch(batch, ctx)

        for X, y in zip(features, labels):
            y = y.astype('float16')

            acc += (net(X).argmax(axis=1) == y).sum().copyto(mxnet.cpu())

            n += y.size

        acc.wait_to_read()

    return acc.asscalar() / n


def train(train_data, test_data, batch_size, net, loss, trainer, ctx, num_epochs, print_batches=None, is_batch=True):
    """Train and evaluate a model."""

    print("training on", ctx)

    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    test_iter = load.get_iter(test_data, batch_size, shuffle=False)
    for epoch in range(1, num_epochs + 1):

        train_iter = load.get_iter(train_data, batch_size, shuffle=is_batch)

        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0

        start = time.time()
        i = 0
        for batch in train_iter:
            i += 1
            t1 = time.time()
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []

            with autograd.record():

                y_hats = [net(X) for X in Xs]

                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]

            for l in ls:
                l.backward()

            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar()

                                  for y_hat, y in zip(y_hats, ys)])

            train_l_sum += sum([l.sum().asscalar() for l in ls])

            trainer.step(batch_size)

            n += batch_size

            m += sum([y.size for y in ys])

            if print_batches and (i + 1) % print_batches == 0:
                print("batch %d, loss %f, train acc %f, %.2fs per batch" % (

                    n, train_l_sum / n, train_acc_sum / m, time.time() - t1

                ))
        tt = time.time()
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('test_time',time.time()-tt)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" % (

            epoch, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start

        ))
        net.save_parameters('param')
        net.save_parameters("test_acc_%.3f_train_acc_%.3f-param" % (

            test_acc, train_acc_sum / m))


net = model.get_netD()
ctx = get_ctx()
net.initialize(force_reinit=True, init=init.Xavier(), ctx=ctx)
net.cast("float16")
if os.path.exists('param'):
    net.load_parameters('param', ctx=ctx)

loss = gluon.loss.SoftmaxCrossEntropyLoss()
batch_size = 30

scheduler = mxnet.lr_scheduler.FactorScheduler(100, 0.9)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.01,
                        'wd': 2e-4,
                        'lr_scheduler': scheduler,
                        'momentum': 0.9,
                        'multi_precision': True})
train_data = load.load_bench()
test_data = load.load_bench()
train(train_data, test_data, batch_size, net, loss, trainer, ctx, 3, 10)
