from DatasetDeepHDR import DatatsetDeepHDR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import os

class DeepHDRNet(nn.Module):
    def __init__(self, OutChannels):
        super().__init__()
        self.OutChannels = OutChannels
        # deined model as described in paper 
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels  = 18, out_channels = 100, kernel_size = (7,7)),
            nn.ReLU(),
            nn.Conv2d(in_channels  = 100, out_channels = 100, kernel_size = (5,5)),
            nn.ReLU(),
            nn.Conv2d(in_channels  = 100, out_channels = 50, kernel_size = (3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels  = 50, out_channels = self.OutChannels, kernel_size = (1,1)),
            nn.Sigmoid()
        )
        # xavier initialization
        init_weight = nn.init.xavier_uniform#normal
        for Layer in self.cnn_layers:
            if isinstance(Layer, nn.Conv2d):
                # print(Layer.weight)
                init_weight(Layer.weight)
                # print(Layer.weight)
    # minimimize tone mapped output values
    def ToneMapNetOut(self, Out):
        Mu = 5000
        Out = torch.log(1 + Mu * Out) / math.log10(1 + Mu)
        return Out
    
    def forward(self, x):
        NetOut = self.cnn_layers(x)
        # WE method
        if self.OutChannels == 9:
            I1 = x[:, 0:3, 6:-6, 6:-6]
            I2 = x[:, 3:6, 6:-6, 6:-6]
            I3 = x[:, 6:9, 6:-6, 6:-6]

            W1 = NetOut[:, 0:3]
            W2 = NetOut[:, 3:6]
            W3 = NetOut[:, 6:9]
            
            PredictedHDR = (W1 * I1 + W2 * I2 + W3 * I3) / (W1 + W2 + W3 + 1e-3)
            # PredictedHDR = self.ToneMapNetOut(PredictedHDR)
        # Direct method
        elif self.OutChannels == 3:
            PredictedHDR = NetOut#self.ToneMapNetOut(NetOut)
        return PredictedHDR
