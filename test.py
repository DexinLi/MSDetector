import os
from mxnet import io, ndarray
import numpy
from struct import unpack
import time

def load(path):
    res = []
    with open(path, 'r') as f:
        for line in f.readlines():
            for s in line.split(" "):
                if s == '??':
                    res.append(256)
                else:
                    res.append(int(s, 16))
    return res

file = "trainLabels.csv"
test = []
train = []
with open(file, "r") as f:
    for idx, line in enumerate(f.readlines()):
        line = line.split(',')
        name = line[0][1:-1]
        label = int(line[1])
        name = '../train/'+name+'.bytes'
        res = load(name)
        if len(res)>1024*1024:
            print(len(res)-1024*1024)
        # if idx%10 >=8:
        #     test.append(())