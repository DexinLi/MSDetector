import os
import mxnet
from struct import unpack
import time

fileLimit = 4*1024*1024

def load(path):
    res = [256]*fileLimit
    i = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            for s in line.split(" "):
                if s[0] != '?':
                    res[i]=(int(s, 16))
                i+=1
    return res

def loadpath():
    train = []
    test = []
    with open('2.txt') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0]
            label = int(line[1])
            name = '../train/'+name+'.bytes'
            if idx%5 == 0:
                test.append((name,label))
            else:
                train.append((name, label))
    return train,test

class Dataset(object):
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, i):
        path = self.dataset[i][0]
        data = mxnet.ndarray.array(load(path),dtype='float16')
        return data, self.dataset[i][1]
    def __len__(self):
        return len(self.dataset)

def get_iter(dataset, batch_size):
    from multiprocessing import cpu_count
    CPU_COUNT = cpu_count()
    d_set = Dataset(dataset)
    dataloader = mxnet.gluon.data.DataLoader(d_set,shuffle=True, batch_size=batch_size,num_workers=CPU_COUNT)
    return dataloader
