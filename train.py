import time
import os
import torch
from torch import nn, optim
import numpy
import random
import load
import model
import sys

batch_mode = True
if len(sys.argv) > 1:
    batch_mode = False

device = torch.device("cuda:0")


def evaluate_accuracy(data_iter, net):
    """Evaluate accuracy of a model on the given data set."""

    acc = 0

    n = 0
    with torch.no_grad():
        for features, labels in data_iter:
            features = features.to(device)
            labels = labels.to(device)
            output = net(features)
            _, ys = torch.max(output.data, 1)
            n += labels.size(0)
            acc += (ys == labels).sum().item()
    return acc / n


def train(train_data, test_data, batch_size, net, loss, trainer, num_epochs, print_batches=None, is_batch=True):
    """Train and evaluate a model."""

    test_iter = load.get_iter(test_data, batch_size, shuffle=False)
    for epoch in range(1, num_epochs + 1):

        train_iter = load.get_iter(train_data, batch_size, shuffle=is_batch)

        train_l_sum, train_acc_sum, n, m = 0.0, 0.0, 0.0, 0.0

        start = time.time()
        i = 0
        for Xs, ys in train_iter:
            i += 1
            t1 = time.time()
            Xs = Xs.to(device)
            ys = ys.to(device)
            y_hats = net(Xs)

            ls = loss(y_hats, ys)

            ls.backward()

            train_acc_sum += (torch.max(y_hats, 1) == ys).sum().item()

            train_l_sum += ls.sum().item()

            trainer.step()

            n += batch_size

            m += batch_size

            if print_batches and (i + 1) % print_batches == 0:
                print("batch %d, loss %f, train acc %f, %ss per batch" % (

                    n, train_l_sum / n, train_acc_sum / m, time.time() - t1

                ))
        tt = time.time()
        test_acc = evaluate_accuracy(test_iter, net)
        print('test_time', time.time() - tt)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" % (

            epoch, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start

        ))
        torch.save(net.state_dict(), 'param.pt')
        torch.save(net.state_dict(), "test_acc_%.3f_train_acc_%.3f-param.pt" % (

            test_acc, train_acc_sum / m))


net = model.Malconv()
if os.path.exists("param.pt"):
    net.load_state_dict(torch.load("param.pt"))

net = nn.DataParallel(net)
net.to(device)

loss = nn.CrossEntropyLoss()
batch_size = 30
test_data = load.loadtest()
if batch_mode:
    trainer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=2e-4)
    train_data = load.loadbatch()

    train(train_data, test_data, batch_size, net, loss, trainer, 10, 10)
else:
    trainer = optim.SGD(net.parameters(), lr=0.008, momentum=0.9, weight_decay=2e-4)
    train_data = load.loadinc()

    train(train_data, test_data, batch_size, net, loss, trainer, 1, 10, False)
