import imageio
import skimage.color
import skimage.io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import utils
import torch
import torchvision
import torch.nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class DatatsetDeepHDR(Dataset):
    def __init__(self, AlignedImagesLDRDir, GtResultsRootDir, Transform = None):
        # List data folders; Aligned images contain 
        self.ListAlignedImagesLDRDir = glob.glob(AlignedImagesLDRDir + "*.npy")
        self.ListAlignedImagesLDRDir.sort()
        self.ListGtResultsDir = glob.glob(GtResultsRootDir + "*/")
        self.ListGtResultsDir.sort()

        if len(self.ListAlignedImagesLDRDir) != len(self.ListGtResultsDir):
            print(len(self.ListGtResultsDir), len(self.ListAlignedImagesLDRDir))
            raise ValueError('Number of ground truth images and aligned input images are not equal')

        self.Transform = Transform
   
    def __getitem__(self, Index):
        AlignedImagesLDR = np.load(self.ListAlignedImagesLDRDir[Index])
        ExpFile = open(self.ListGtResultsDir[Index] + "exposure.txt")
        Exposures = ExpFile.readlines()
        Exposures = 2 ** (np.array(Exposures).astype(float))
        AlignedImagesHDR = utils.LDRToHDR(AlignedImagesLDR, Exposures)
        AlignedImagesLDR = np.concatenate((AlignedImagesLDR[0], AlignedImagesLDR[1], AlignedImagesLDR[2]), axis = -1)
        AlignedImagesHDR = np.concatenate((AlignedImagesHDR[0], AlignedImagesHDR[1], AlignedImagesHDR[2]), axis = -1)
        InpImages = np.concatenate((AlignedImagesLDR, AlignedImagesHDR), axis = -1)
        # InpImages = AlignedImagesLDR
        InpImages = np.transpose(InpImages, (2, 0, 1))

        HDRImage = np.load(self.ListGtResultsDir[Index] + "HDRImg.npy")
        # HDRImage = utils.ToneMapImage(HDRImage)
        HDRImage = np.transpose(HDRImage, (2, 0, 1))

        if self.Transform is not None:
            InpImages = self.transform(InpImages)
            HDRImage = self.transform(HDRImage)
        else:
            InpImages = torch.tensor(InpImages)
            HDRImage = torch.tensor(HDRImage)
        
        return InpImages, HDRImage

    def __len__(self):
        return len(self.ListAlignedImagesLDRDir)

# if __name__ == "__main__":
#     d=DatatsetDeepHDR("../data/AlignedImagesTrain/", "../data/Trainingset/Training/")
#     d.__getitem__(10)
    

# for BatchNum in range(PredictedHDR.shape[0]):
#             for ColorChannel in range(3):
#                 PredictedHDR[BatchNum, :, :, ColorChannel] = utils.Normalize(PredictedHDR[BatchNum, :, :, ColorChannel], \
#                     torch.min(PredictedHDR[BatchNum, :, :, ColorChannel]), torch.max(PredictedHDR[BatchNum, :, :, ColorChannel]))