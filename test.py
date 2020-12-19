from NetDeepHDR import DeepHDRNet
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import os
import glob
import numpy as np
import sys
sys.path.append('../pyflow/')
import pyflow
import time


TestInputDataDir = '../data/Testset/Test/004/'
# MyTrainedModelPath = '../models/models_aws/ModelRepMainWe1.pth'
MyTrainedModelPath = '../models/models_aws/All18We_Ihvhighhopes1.pth'
UseCuda = torch.cuda.is_available()
Device = torch.device("cuda" if UseCuda else "cpu")
NumOut = 9
GroundTruthExists = True

def CeLiuOpticalFlow(Image1, Image2, ExposureRestoreFactor):
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY

    u, v, WarpedIm2 = pyflow.coarse2fine_flow(Image1, Image2, alpha, ratio, minWidth, 
        nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    WarpedIm2 = WarpedIm2*ExposureRestoreFactor
    plt.imshow(u)
    plt.show()
    plt.imshow(v)
    plt.show()
    return WarpedIm2

# Load pre-trained model of official implementation
Model = DeepHDRNet(NumOut)
Model = Model.to(Device)
Optimizer = torch.optim.Adam(Model.parameters(), 1e-5)
Model, Optimizer = load_checkpoint(MyTrainedModelPath, Model, Optimizer)

TestFlowDataList = glob.glob(TestInputDataDir + "*.tif" )
TestFlowDataList.sort()
Image1 = ReadRawImage(TestFlowDataList[0])
Image2 = ReadRawImage(TestFlowDataList[1])
Image3 = ReadRawImage(TestFlowDataList[2])
Image1 = Normalize(Image1, np.min(Image1), np.max(Image1))
Image2 = Normalize(Image2, np.min(Image2), np.max(Image2))
Image3 = Normalize(Image3, np.min(Image3), np.max(Image3))

ExpFile = open(TestInputDataDir + "exposure.txt")
Exposures = ExpFile.readlines()
Exposures = np.array(Exposures).astype(float)
# print(Exposures)

# Normalize input images
Exposure1 = (2 ** (Exposures[1] - Exposures[0])) ** (1 / 2.2)
Exposure3 = (2 ** (Exposures[1] - Exposures[2])) ** (1 / 2.2)
Image1_e = Exposure1 * Image1
Image3_e = Exposure3 * Image3

AlignedImagesLDR = []
Im1w = CeLiuOpticalFlow(Image2, Image1_e, 1 / Exposure1)
# SaveImage(Im1w, "temptest/" + "%03d"%(Idx) + "_1.png")
AlignedImagesLDR.append(Im1w.copy())
AlignedImagesLDR.append(Image2.copy())
Im3w = CeLiuOpticalFlow(Image2, Image3_e, 1 / Exposure3)
AlignedImagesLDR.append(Im3w.copy())
AlignedImagesLDR = np.array(AlignedImagesLDR)

SaveImage(Im1w, 'temp1.png')
SaveImage(Im3w, 'temp2.png')

start = time.time()
# SaveImage(Im3w, "temp/" + "%03d"%(Idx) + "_3.png")
# np.save(DefaultOutputFolder + "%03d"%(Idx) + ".npy", Results)
Exposures = 2 ** (np.array(Exposures).astype(float))
AlignedImagesHDR = LDRToHDR(AlignedImagesLDR, Exposures)
AlignedImagesLDR = np.concatenate((AlignedImagesLDR[0], AlignedImagesLDR[1], AlignedImagesLDR[2]), axis = -1)
AlignedImagesHDR = np.concatenate((AlignedImagesHDR[0], AlignedImagesHDR[1], AlignedImagesHDR[2]), axis = -1)
InpImages = np.concatenate((AlignedImagesLDR, AlignedImagesHDR), axis = -1)
InpImages = np.transpose(InpImages, (2, 0, 1))

TestInput = torch.tensor(InpImages)
TestInput = torch.unsqueeze(TestInput, 0)
TestInput = TestInput.float().to(Device)
Model.eval()
with torch.no_grad():
    Prediction = Model(TestInput)

# if NumOut == 9:
#     print("FML")
#     print(Prediction.shape)
#     I1 = TestInput[:, 0:3, 6:-6, 6:-6].cpu().numpy()
#     I2 = TestInput[:, 3:6, 6:-6, 6:-6].cpu().numpy()
#     I3 = TestInput[:, 6:9, 6:-6, 6:-6].cpu().numpy()

#     W1 = Prediction[:, 0:3].cpu().numpy()
#     W2 = Prediction[:, 3:6].cpu().numpy()
#     W3 = Prediction[:, 6:9].cpu().numpy()

#     Prediction = (W1 * I1 + W2 * I2 + W3 * I3) / (W1 + W2 + W3 + 1e-3)

Prediction = Prediction[0].cpu().numpy()
Prediction = np.transpose(Prediction, (1, 2, 0))
Prediction = Normalize(Prediction, np.min(Prediction), np.max(Prediction))
# Prediction = ToneMapImage(Prediction)
plt.imshow(Prediction)
plt.show()
if GroundTruthExists:
    HDRImg = np.load(TestInputDataDir + "HDRImg.npy")
    HDRImg = HDRImg[6:-6,6:-6]
    HDRImg = ToneMapImage(HDRImg)
    Mse = np.mean((Prediction - HDRImg) ** 2)
    plt.imshow(HDRImg)
    plt.show()
    print(Mse)
    print(time.time() - start)