import os
import torch
from torch.utils.data import DataLoader
import time
import numpy

from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
fileLimit = 4*1024*1024
SPLIT = 5

def load(path):
    res = numpy.full(fileLimit, 257)
    i = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            for idx, s in enumerate(line.split()):
                if idx == 0:
                    continue
                if s != '??':
                    res[i] = (int(s, 16))
                else:
                    res[i] = 256
                i+=1
    return res

def isTest(idx):
    return idx % 5 == 0

def loadbatch():
    train = []
    with open('2.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = '../train/'+name+'.bytes'
            if label > SPLIT:
                break
            if not isTest(idx):
                train.append((name,label))
    return train

def loadtest():
    test = []
    with open('2.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = '../train/'+name+'.bytes'
            if isTest(idx):
                test.append((name,label))
    return test

def loadinc():
    train = []
    with open('2.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = '../train/'+name+'.bytes'
            if not isTest(idx):
                train.append((name, label))
    return train

def load_bench():
    res = []
    with open("bench.txt") as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = './train/' + name + '.bytes'
            res.append((name, label))
    return res

class Dataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, i):
        path = self.dataset[i][0]
        data = load(path)
        label = torch.full([10], 0,dtype=torch.long)
        label[self.dataset[i][1]] = 1
        return data, label
    def __len__(self):
        return len(self.dataset)

def get_iter(dataset, batch_size, shuffle=True):
    d_set = Dataset(dataset)
    dataloader = DataLoader(d_set,shuffle=shuffle, batch_size=batch_size,num_workers=CPU_COUNT)
    return dataloader
