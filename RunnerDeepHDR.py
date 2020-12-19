from DatasetDeepHDR import DatatsetDeepHDR
from NetDeepHDR import DeepHDRNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import os


class RunnerDeepHDR(object):
    def __init__(self, TrainAlignedImagesLDRDir, TrainGtResultsRootDir, TestAlignedImagesLDRDir, TestGtResultsRootDir, NetworkModel, BatchSize, NumEpochs, lr, Device, ModelStorePath, UsePrevModel = False):
        # @todo: set up transform 

        # setup parameters for logging and device
        self.NumEpochs = NumEpochs
        self.LogFreq = 20  # Steps
        self.ValFreq = 20  # Steps
        self.Device = Device
        self.Writer = SummaryWriter()
        self.ModelStorePath = ModelStorePath
        
        # Import dataloader 
        TrainDataset = DatatsetDeepHDR(TrainAlignedImagesLDRDir, TrainGtResultsRootDir)
        self.TrainDatasetLoader = DataLoader(TrainDataset, batch_size=BatchSize, shuffle=True, num_workers=4)

        TestDataset = DatatsetDeepHDR(TestAlignedImagesLDRDir, TestGtResultsRootDir)
        self.TestDatasetLoader = DataLoader(TestDataset, batch_size=BatchSize, shuffle=True, num_workers=4)
        
        
        # Load model
        NumOut = 3
        if NetworkModel == "WE":
            NumOut = 9
        self.Model = DeepHDRNet(NumOut)
        self.Model = self.Model.to(self.Device)

        # Set optimizer
        self.Optimizer = torch.optim.Adam(self.Model.parameters(), lr)
        # self.Optimizer = torch.optim.SGD(self.Model.parameters(), lr, momentum = 0.9)
        self.LossCriterion = nn.MSELoss(reduction = "sum")

        # load pretrained model 
        if UsePrevModel:
            utils.load_checkpoint(self.ModelStorePath, self.Model, self.Optimizer)   

    def Optimize(self, Prediction, Label):
        # Calculate Loss
        Loss = self.LossCriterion(Prediction, Label)
        # Update weights
        self.Optimizer.zero_grad()
        Loss.backward()
        self.Optimizer.step()
        return Loss

    def Validate(self):
        self.Model = self.Model.to(self.Device)
        self.Model.eval()
        TotalValLoss = 0.0
        TotalPSNR = 0.0
        with torch.no_grad():
            for Images, Label in self.TestDatasetLoader:
                Images, Label = Images.float().to(self.Device), Label.float().to(self.Device)
                Label = Label[:, :, 6:-6, 6:-6]
                Prediction = torch.squeeze(self.Model(Images))
                Loss = self.LossCriterion(Prediction, Label)
                TotalValLoss += Loss.item()
                TotalPSNR += utils.EstimatePSNR(Prediction, Label)
        TotalValLoss /= len(self.TestDatasetLoader)
        TotalPSNR /= len(self.TestDatasetLoader)
        # Prediction = Prediction[0].cpu().numpy()
        # Prediction = np.transpose(Prediction, (1, 2, 0))
        # print(Prediction[:,:,0])
        return TotalValLoss, TotalPSNR

    def Train(self):
        NumBatches = len(self.TrainDatasetLoader)
        PrevLoss = 1e10
        for Epoch in range(self.NumEpochs):
            for BatchItrIndex, (Images, Label) in enumerate(self.TrainDatasetLoader):
                self.Model.train()
                Images, Label = Images.float().to(self.Device), Label.float().to(self.Device)
                Label = Label[:, :, 6:-6, 6:-6]
                Prediction = torch.squeeze(self.Model(Images))
                Loss = self.Optimize(Prediction, Label)
                
                CurrentStep = Epoch * NumBatches + BatchItrIndex
                if CurrentStep % self.LogFreq == 0:
                    # Visualize
                    print("Epoch: {}, Batch {}/{} has loss {}".format(Epoch, BatchItrIndex, NumBatches, Loss))
                    self.Writer.add_scalar('Train/Loss', Loss, CurrentStep)
                    PSNR = utils.EstimatePSNR(Prediction, Label)
                    self.Writer.add_scalar('Train/PSNR', PSNR, CurrentStep)
                    #Save model
                    # if Loss.item() < PrevLoss:
                    utils.save_checkpoint(self.ModelStorePath, self.Model, self.Optimizer)
                        #PrevLoss = Loss.item()
                
                if CurrentStep % self.ValFreq == 0:
                    # Validate
                    self.Model.eval()
                    ValLoss, PSNR = self.Validate()
                    self.Model.train()
                    # Visualize
                    self.Writer.add_scalar('Validation/Loss', ValLoss, CurrentStep)
                    self.Writer.add_scalar('Validation/PSNR', PSNR, CurrentStep)