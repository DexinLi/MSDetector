from mxnet.gluon import nn
import mxnet


class GLU(nn.HybridBlock):
    def __init__(self, channels, kernel_size, stride):
        super(GLU, self).__init__()
        self.conv = nn.Conv1D(channels, kernel_size, stride)
        self.channels = channels

    def hybrid_forward(self, F, X, *args, **kwargs):
        X = X.transpose((0, 2, 1))
        X = self.conv(X)
        channels = self.channels // 2
        Y1 = mxnet.nd.slice_axis(X, axis=1, begin=0, end=channels)
        Y2 = mxnet.nd.slice_axis(X, axis=1, begin=channels, end=self.channels)
        Z = mxnet.nd.multiply(Y1, mxnet.nd.sigmoid(Y2))
        Z = Z.transpose((0, 2, 1))
        return Z.reshape((0, 1, -1))


def get_netD():
    netD = nn.Sequential()
    netD.add(nn.Embedding(128, 7, dtype='float16'),
             GLU(channels=128, kernel_size=64, stride=64),
             nn.MaxPool1D(64, 64),
             nn.Dense(128, activation="sigmoid"),
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