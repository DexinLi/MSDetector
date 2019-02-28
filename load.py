import os
import mxnet
import pickle

from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
fileLimit = 512*1024
SPLIT = 5

def load(path):
    res = []
    i = 0
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res + (fileLimit - len(res)) * [127]

def isTest(idx):
    return idx % 5 == 0

def loadbatch():
    train = []
    with open('2.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = './op/'+name+'.pickle'
            if label > SPLIT:
                break
            if not isTest(idx):
                train.append((name,label))
    return train

def batch_test():
    test = []
    with open('2.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = './op/'+name+'.pickle'
            if label <= SPLIT and isTest(idx):
                test.append((name,label))
    return test

def inc_test():
    test = []
    with open('2.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = './op/' + name + '.pickle'
            if isTest(idx):
                test.append((name, label))
    return test

def loadinc():
    train = []
    with open('2.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = './op/'+name+'.pickle'
            if label <= SPLIT:
                continue
            if not isTest(idx):
                train.append((name, label))
    return train



class Dataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, i):
        path = self.dataset[i][0]
        data = mxnet.ndarray.array(load(path),dtype='float16')
        return data, self.dataset[i][1]
    def __len__(self):
        return len(self.dataset)

def get_iter(dataset, batch_size, shuffle=True):
    d_set = Dataset(dataset)
    dataloader = mxnet.gluon.data.DataLoader(d_set,shuffle=shuffle, batch_size=batch_size,num_workers=CPU_COUNT)
    return dataloader
