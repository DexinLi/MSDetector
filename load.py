import os
import multiprocessing
from multiprocessing import Process,Manager
from struct import unpack
import time

fileLimit = 4*1024*1024

def load(path):
    res = [256]*fileLimit
    i = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            for s in line.split(" "):
                if i >= fileLimit:
                    return None
                if s[0] != '?':
                    res[i]=(int(s, 16))
                i+=1
    return res

def loadpath():
    train = []
    test = []
    with open('trainLabels.csv') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.split(',')
            name = line[0][1:-1]
            label = int(line[1])
            name = '../train/'+name+'.bytes'
            if idx%5 == 0:
                test.append((name,label))
            else:
                train.append((name, label))
    return train,test


def get_iter(dataset, batch_size):
    n = len(dataset)
    res = []
    for i in range(n // batch_size):
        x = []
        y = []
        for j in range(batch_size):
            data = dataset[i * batch_size + j]
            x.append(data[0])
            y.append(data[1])
        res.append((x, y))
    return res
