from torch import nn
import torch

class Malconv(nn.Module):
    def __init__(self):
        super(Malconv, self).__init__()
        self.embed = nn.Embedding(258, 8)
        self.conv = nn.Conv1d(8, 128, kernel_size=512, stride=512)
        self.glu = nn.GLU()
        self.pooling = nn.MaxPool1d(64, 64)
        self.fc = nn.Linear(8192, 128)
        self.sig = nn.Sigmoid()
        self.out = nn.Linear(128, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embed(x)
        x = torch.transpose(x, -1, -2)
        x = self.conv(x)
        x = self.glu(x)
        x = torch.transpose(x, -1, -2)
        x = torch.Tensor()
        x = x.view((x.shape[0], 1, -1))
        x = self.pooling(x)
        x = self.fc(x)
        x = self.sig(x)
        x = self.out(x)
        return self.softmax(x)