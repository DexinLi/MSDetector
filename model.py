from mxnet.gluon import nn
import mxnet


class GLU(nn.HybridBlock):
    def __init__(self, channels, kernel_size, stride):
        super(GLU, self).__init__()
        self.conv = nn.Conv1D(channels, kernel_size, stride)

    def hybrid_forward(self, F, X, *args, **kwargs):
        X = X.transpose((0, 2, 1))
        X = self.conv(X)
        Y1, Y2 = F.split(X, axis=1, num_outputs=2)
        Z = F.elemwise_mul(Y1, F.sigmoid(Y2))
        Z = Z.transpose((0, 2, 1))
        return Z.reshape((0, 1, -1))

class GLU0(nn.HybridBlock):
    def __init__(self, channels, kernel_size, stride):
        super(GLU0, self).__init__()
        self.conv1 = nn.Conv1D(channels, kernel_size, stride)
        self.conv2 = nn.Conv1D(channels, kernel_size, stride)
    
    def hybrid_forward(self, F, X, *args, **kwargs):
        X = X.transpose((0, 2, 1))
        Y1 = self.conv1(X)
        Y2 = self.conv2(X)
        Z = F.elemwise_mul.multiply(Y1, F.sigmoid(Y2))
        Z = Z.transpose((0, 2, 1))
        return Z.reshape((0, 1, -1))

def get_netD():
    netD = nn.HybridSequential()
    netD.add(nn.Embedding(128, 7, dtype='float16'),
             GLU(channels=128, kernel_size=4, stride=1),
             nn.MaxPool1D(64, 64),
             nn.Dense(2048, activation="sigmoid"),
             nn.Dense(10))
    return netD

def get_netD0():
    netD = nn.HybridSequential()
    netD.add(nn.Embedding(128, 7, dtype='float16'),
             GLU0(channels=128, kernel_size=4, stride=1),
             nn.MaxPool1D(128, 128),
             nn.Dense(2048, activation="sigmoid"),
             nn.Dense(10))
    return netD

def get_netD1():
    netD = nn.Sequential()
    netD.add(nn.Conv1D(channels=8, kernel_size=4, strides=1, activation='relu'),
             nn.MaxPool1D(pool_size=4, strides=1),
             nn.Conv1D(channels=128, kernel_size=512, strides=512, activation='relu'),
             nn.MaxPool1D(pool_size=4, strides=4),
             nn.Conv1D(channels=256, kernel_size=4, strides=4, activation='relu'),
             nn.MaxPool1D(pool_size=4, strides=4),
             nn.Dense(128),
             nn.Dense(10))
    return netD