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

DefaultOutputFolder = "../data/AlignmentImagesTest/"
os.makedirs(DefaultOutputFolder, exist_ok = "true")
DefaultInputFolder = "../data/Testset/Test/"

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
    '''
    NumRows = Image1.shape[0]
    NumCols = Image1.shape[1]
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
    '''
    return WarpedIm2

if __name__ == "__main__":
    if os.path.exists(DefaultInputFolder):
        InpFolders = glob.glob(DefaultInputFolder + "*/")
        InpFolders.sort()
        for Idx, Folder in enumerate(InpFolders):
            print("Processing Idx, Folder ", Idx, Folder)
            InpTiffFiles = glob.glob(Folder + "*.tif" )
            InpTiffFiles.sort()
            # read and normalize images
            Image1 = ReadRawImage(InpTiffFiles[0])
            Image2 = ReadRawImage(InpTiffFiles[1])
            Image3 = ReadRawImage(InpTiffFiles[2])
            Image1 = Normalize(Image1, np.min(Image1), np.max(Image1))
            Image2 = Normalize(Image2, np.min(Image2), np.max(Image2))
            Image3 = Normalize(Image3, np.min(Image3), np.max(Image3))

            ExpFile = open(Folder + "exposure.txt")
            Exposures = ExpFile.readlines()
            Exposures = np.array(Exposures).astype(float)
            print(Exposures)
            
            # Normalize input images
            Exposure1 = (2 ** (Exposures[1] - Exposures[0])) ** (1 / 2.2)
            Exposure3 = (2 ** (Exposures[1] - Exposures[2])) ** (1 / 2.2)
            Image1_e = Exposure1 * Image1
            Image3_e = Exposure3 * Image3

            Results = []
            Im1w = CeLiuOpticalFlow(Image2, Image1_e, 1 / Exposure1)
            SaveImage(Im1w, "../temptest/" + "%03d"%(Idx) + "_1.png")
            Results.append(Im1w.copy())
            Results.append(Image2.copy())
            Im3w = CeLiuOpticalFlow(Image2, Image3_e, 1 / Exposure3)
            Results.append(Im3w.copy())
            SaveImage(Im3w, "../temptest/" + "%03d"%(Idx) + "_3.png")
            np.save(DefaultOutputFolder + "%03d"%(Idx) + ".npy", Results)

            # if Idx > 8:
            #     break
    
    # if os.path.exists(DefaultInputFolder):
    #     InpFolders = glob.glob(DefaultInputFolder + "*/")
    #     InpFolders.sort()
    #     for Idx, Folder in enumerate(InpFolders):
    #         print("Processing Idx, Folder ", Idx, Folder)
    #         InpHDR = skimage.io.imread(Folder + "HDRImg.hdr")
    #         np.save(Folder + "HDRImg" + ".npy", InpHDR.copy())
    


    
