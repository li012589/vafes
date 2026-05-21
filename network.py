import torch
from torch import nn


class ResNet1d(nn.Module):
    def __init__(self, mlpVector, channel, kernelSize, outputChannels, hiddenChannels, hiddenConvLayers, hiddenWidth, hiddenFcLayers, activation=nn.ReLU()):
        super().__init__()
        mlpList = []
        for no in range(len(mlpVector)-1):
            mlpList.append(nn.Linear(mlpVector[no], mlpVector[no+1]))
        self.mlp = nn.ModuleList(mlpList)

        assert mlpVector[-1] % channel == 0
        self.channel = channel
        self.firstConv = nn.Conv1d(channel, 2 * hiddenChannels, kernelSize, padding=(kernelSize - 1) // 2)

        self.hiddenConv = []
        for _ in range(hiddenConvLayers):
            self.hiddenConv.append(nn.Sequential(
                nn.Conv1d(2 * hiddenChannels, hiddenChannels, 1),
                activation,
                nn.Conv1d(hiddenChannels, hiddenChannels, kernelSize, padding=(kernelSize - 1) // 2),
                activation,
                nn.Conv1d(hiddenChannels, 2 * hiddenChannels, 1),
            ))
        self.hiddenConv = nn.ModuleList(self.hiddenConv)

        self.firstFc = nn.Conv1d(2 * hiddenChannels, hiddenWidth, 1)
        self.hiddenFc = []
        for _ in range(hiddenFcLayers):
            self.hiddenFc.append(
                nn.Conv1d(hiddenWidth, hiddenWidth, 1)
            )
        self.hiddenFc = nn.ModuleList(self.hiddenFc)

        self.finalFc = nn.Conv1d(hiddenWidth, outputChannels, 1)

        self.activation = activation

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
            x = self.activation(x)

        x = x.reshape(x.shape[0], self.channel, -1)
        x = self.firstConv(x)
        x = self.activation(x)

        for layer in self.hiddenConv:
            tmp = x
            tmp = layer(tmp)
            tmp = self.activation(tmp)
            x = x + tmp

        x = self.firstFc(x)
        x = self.activation(x)

        for layer in self.hiddenFc:
            x = layer(x)
            x = self.activation(x)

        x = self.finalFc(x)
        return x
