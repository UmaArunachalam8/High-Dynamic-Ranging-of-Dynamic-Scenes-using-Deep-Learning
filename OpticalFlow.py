from utils import *
import scipy.interpolate
import scipy.ndimage
from matplotlib import pyplot as plt
import numpy as np 
import os
import sys
import glob
sys.path.append('../pyflow/')
import pyflow

TestDataPath = "../Reference/OpticalFlowData/"
DefaultOpFolder = TestDataPath + "Pyflow/LadySitting/"
os.makedirs(DefaultOpFolder, exist_ok = "true")
DefaultOutputFolder = "../op_/"
os.makedirs(DefaultOutputFolder, exist_ok = "true")

DefaultInputFolder = "../OfficialDeepHDR/Trainingset/Training/"
dsf = 10

def CeLiuOpticalFlow(Image1, Image2, ExposureRestoreFactor, SaveName):
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

    NumRows = Image1.shape[0]
    NumCols = Image1.shape[1]

    '''
    x, y = np.meshgrid(np.arange(NumCols), np.arange(NumRows))
    f = scipy.interpolate.interp2d(x[::10, ::10], y[::10, ::10], Image2[::10, ::10, 0])
    x_ = x + u
    y_ = y + v
    WarpedIm2_ = f(x_[::10], y_[::10])
    '''
    
    # N = NumRows * NumCols
    # T = np.zeros((3, 3 * N))
    # T[:2, ::3] = 1
    # T[2, 2::3] = 1  
    # T[0, 2::3] = -1 * u.flatten()
    # T[1, 2::3] = -1 * v.flatten()
    
    x, y = np.meshgrid(np.arange(NumCols), np.arange(NumRows))
    x_ = x + u
    y_ = y + v 

    x_ = x_.astype(int)
    y_ = y_.astype(int)

    x_ = np.where((x_ < 0) | (x_ >= NumCols), 0, x_)
    y_ = np.where((y_ < 0) | (y_ >= NumRows), 0, y_)

    ImgWarped = np.zeros_like(Image1)
    ImgWarped[y.flatten(), x.flatten()] = Image2[y_.flatten(), x_.flatten()]
    ImgWarped *= ExposureRestoreFactor
    
    
    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(ImgWarped / ExposureRestoreFactor)
    # plt.axis('off')
    # plt.title("from u v manual")
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(Image1)
    # plt.axis('off')
    # plt.title("reference image")
    # plt.savefig(SaveName)
    # plt.show()

    SaveImage(WarpedIm2, SaveName + ".png")
    SaveImage(ImgWarped, SaveName + "my.png")
    return WarpedIm2

if __name__ == "__main__":
    Image1 = ReadRawImage(TestDataPath + "3.tif")
    Image2 = ReadRawImage(TestDataPath + "4.tif")
    Image3 = ReadRawImage(TestDataPath + "5.tif")
    Image1 = Normalize(Image1, np.min(Image1), np.max(Image1))
    Image2 = Normalize(Image2, np.min(Image2), np.max(Image2))
    Image3 = Normalize(Image3, np.min(Image3), np.max(Image3))
    SaveImage(Image1, "Image1.png")
    SaveImage(Image2, "Image2.png")
    SaveImage(Image3, "Image3.png")

    
    # Normalize input images
    # Image1_e = 2 * Image1
    # Image3_e = Image3 / 2
    Image1_e = (2 ** (2 - 0)) ** (1 / 2.2) * Image1
    Image3_e = (2 ** (2 - 4)) ** (1 / 2.2) * Image3

    Results = []

    Im1w = CeLiuOpticalFlow(Image2, Image1_e, 1 / ((2 ** (2 - 0)) ** (1 / 2.2)), "hi")
    Results.append(Im1w.copy())
    Results.append(Image2.copy())
    Im3w = CeLiuOpticalFlow(Image2, Image3_e, 1 / ((2 ** (2 - 4)) ** (1 / 2.2)), "bye")
    Results.append(Im3w.copy())
    np.save(DefaultOutputFolder + "WarpedImages.npy", Results)
    
    


    
