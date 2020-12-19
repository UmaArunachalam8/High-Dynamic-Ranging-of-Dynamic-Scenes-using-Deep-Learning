#import OpenEXR
import matplotlib.pyplot as plt
import numpy as np 
import skimage.io
import skimage.color
import skimage.transform
# import Imath
import os
from utils import *
import time

DefaultOutputFolder = "../Output/"
DefaultDataDirectory = "../data/" 
os.makedirs(DefaultOutputFolder, exist_ok = True)
NumImages = 3
Dsf = 200 #DownSamplinfFactor
start = time.time()
def ApplyWeighting(Input, Zmin, Zmax, Type = "Tent", ScaleFactor = 1, Exposure = 1, g = 2.16962522, Var = 25.01731058):
    if Type == "Uniform":
        Input = np.where(((Input >= Zmin) & (Input <= Zmax)), 1 * ScaleFactor, 0)
    elif Type == "Tent":
        Indices = ((Input >= Zmin) & (Input <= Zmax)).astype(int)
        Input = np.minimum(Input, 1 * ScaleFactor - Input) * Indices
    elif Type == "Gaussian":
        Mean = 0.5 * (Zmax + Zmin)
        Indices = ((Input >= Zmin) & (Input <= Zmax)).astype(int)
        Input = ScaleFactor * np.exp((-4 * (Input - Mean) ** 2) / (Mean ** 2))
        Input *= Indices
    elif Type == "Photon":
        Input = np.where(((Input >= Zmin) & (Input <= Zmax)), Exposure * 1 * ScaleFactor, 0)
    else:
        Input = np.where(((Input >= Zmin) & (Input <= Zmax)), ScaleFactor * (Exposure ** 2) / (g * Input + Var), 0)
    
    return Input

def LinearizeImages(ImagesList, Zmin, Zmax, Weighting = "Uniform"):
    gDim = Zmax - Zmin + 1
    A = None
    b = None
    for ImageIndex in range(NumImages):
        # Read and sample image locations
        # ImagePath = ImagesDir + "exposure" + str(ImageIndex + 1) + ".jpg"
        # Image = skimage.io.imread(ImagePath)
        ImageRead = ImagesList[ImageIndex]
        Image = ImageRead[::Dsf, ::Dsf].reshape(1, -1)

        # Apply weightting to sampled locations
        Exposure = np.log(2 ** (ImageIndex - 11))
        Weight = ApplyWeighting(Image, Zmin, Zmax, Weighting, 255, np.exp(Exposure))

        # Calculate A and b for least squares
        NumPixels = Image.size
        RowRange = np.arange(NumPixels).reshape(1, -1)
        ATemp = np.zeros((NumPixels, gDim + NumPixels))
        ATemp[RowRange, Image] = Weight[0, :]
        ATemp[RowRange, RowRange + gDim] = -1 * Weight[0, :]

        bTemp = np.ones((NumPixels, 1)) * Exposure * Weight[0, :].reshape(-1, 1)
        
        A = ATemp if ImageIndex == 0 else np.vstack((A, ATemp))
        b = bTemp if ImageIndex == 0 else np.vstack((b, bTemp))

    # Add regularization term
    InitASize = A.shape[0]
    A = np.vstack((A, np.zeros((gDim, gDim + NumPixels))))
    b = np.vstack((b, np.zeros((gDim, 1))))

    A[InitASize, 128] = 1

    SmoothnessReg = np.arange(gDim).reshape(1, -1)
    SmoothnessReg += Zmin
    SmoothnessReg = ApplyWeighting(SmoothnessReg, Zmin, Zmax, Weighting, 255)
    Lambda = 50
    Scale = np.array([1, -2, 1])
    
    for Index in range(0, gDim - 1):
        A[Index + InitASize + 1, Index : Index + 3] = Scale * Lambda * SmoothnessReg[0, Index + 1]
        
    # Solve for Av = b
    v = np.linalg.lstsq(A, b, rcond=None)[0]
    g = v[:gDim, 0]
    
    plt.plot(g, 'b+', markersize = 6)
    if Weighting != "Uniform":
        plt.ylim(-5, 5)
    plt.savefig(DefaultOutputFolder + Weighting + ".png")
    # plt.show()
    plt.cla()
    plt.plot(np.exp(g), 'r+')
    plt.savefig(DefaultOutputFolder + "Exp" + Weighting + ".png")
    # plt.show()
    plt.cla()
    return g

def MergeLDRImages(LDRList, LinList, Zmin = 0.05, Zmax = 0.95, Start = 0, End = 255, Weighting = "Uniform", Merging = "Linear", g = 2.16962522, Var = 25.01731058):
    Nr = np.zeros_like(LDRList[0]) * 1.0 
    Dr = np.zeros_like(LDRList[0]) * 1.0 
    
    for ImageIndex in range(NumImages):
        # LdrImg = Normalize(LDRList[ImageIndex], Start, End)
        # LinImg = Normalize(LinList[ImageIndex], Start, End)
        LdrImg = LDRList[ImageIndex]
        LinImg = LinList[ImageIndex]
        Exposure = 2 ** ImageIndex#(ImageIndex - 11)
        if Weighting == "Optimal":
            W = ApplyWeighting(LdrImg, Zmin, Zmax, Weighting, 1, Exposure, g, Var)
        else:
            W = ApplyWeighting(LdrImg, Zmin, Zmax, Weighting, 1, Exposure)
        Dr += W
        if Merging == "Linear":
            Nr += (W * LinImg / Exposure)
        else:
            Nr += (W * (np.log(LinImg + 1e-1) - np.log(Exposure)))
        
    IHdr = Nr / Dr
    MaxValidPix = np.max(IHdr[np.where((Dr != 0) & (~np.isnan(Nr)))])
    MinValidPix = np.min(IHdr[np.where((Dr != 0) & (~np.isnan(Nr)))])
    # MinValidPix = np.min(IHdr[np.where(Dr != 0)])
    # print(MaxValidPix, MinValidPix)
    #Wmin = np.where(Dr == 0, np.inf, IHdr)
    #MinValidPix = np.min(Wmin)
    
    IHdr = np.where((Dr == 0) & (LdrImg > 0.5), MaxValidPix, IHdr)
    # IHdr = np.where((np.isnan(Nr)) & (LdrImg > 0.5), MaxValidPix, IHdr)
    IHdr = np.where((Dr == 0) & (LdrImg < 0.5), MinValidPix, IHdr)
    # IHdr = np.where((np.isnan(Nr)) & (LdrImg < 0.5), MinValidPix, IHdr)
    IHdr = np.where(np.isnan(Nr), MinValidPix, IHdr)

    if Merging == "Logarithmic":
        IHdr = np.exp(IHdr)

    return IHdr

def ChoosePatchColorChecker(Image, SaveName = "TempLoc"):
    # Image = skimage.io.imread(ImagePath)
    # Normalize(Image, 0, np.max(Image))
    plt.imshow(Image, cmap = 'gray')
    Location = []
    for i in range(1):
        if i == 0:
            plt.waitforbuttonpress()
        pts = plt.ginput(2)
        (x0, y0), (x1, y1) = pts
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        Location.append([xmin, xmax, ymin, ymax])
    print(Location)
    # np.save(SaveName, Location)
    plt.show()

# def OpenExrfile(Path):
#     pt = Imath.PixelType(Imath.PixelType.FLOAT)
#     golden = OpenEXR.InputFile(Path)
#     dw = golden.header()['dataWindow']
#     size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

#     redstr = golden.channel('R', pt)
#     red = np.fromstring(redstr, dtype = np.float32)
#     red.shape = (size[1], size[0]) # Numpy arrays are (row, col)

#     greenstr = golden.channel('G', pt)
#     green = np.fromstring(greenstr, dtype = np.float32)
#     green.shape = (size[1], size[0]) # Numpy arrays are (row, col)

#     bluestr = golden.channel('B', pt)
#     blue = np.fromstring(bluestr, dtype = np.float32)
#     blue.shape = (size[1], size[0]) # Numpy arrays are (row, col)

#     return np.dstack((red, green, blue))

def PhotographicToneMappingOperator(Image):
    B = 0.95
    K = 0.15
    NInv = 1 / Image.size 
    eps = 1e-8
    if Image.ndim == 3:
        NInv *= 3
        eps = 1e-4
        B = 0.95
        K = 0.15
    Im = np.exp(NInv * np.sum(np.log(Image + eps), axis = (0, 1)))
    Ihdr_ = K * Image / Im
    IWhite = B * np.max(Ihdr_, axis = (0, 1))
    ImageToneMapped = (Ihdr_ * (1 + (Ihdr_ / (IWhite * IWhite)))) / (1 + Ihdr_)
    return ImageToneMapped

if __name__ == "__main__":

    # LinearTiffList = np.load(DefaultDataDirectory + "055.npy")
    LinearTiffList = []
    LinearTiffList.append(ReadRawImage(DefaultDataDirectory + 'mine1.tiff') / 255)
    LinearTiffList.append(ReadRawImage(DefaultDataDirectory + 'mine2.tiff') / 255)
    LinearTiffList.append(ReadRawImage(DefaultDataDirectory + 'mine3.tiff') / 255)
    LinearTiffList = np.array(LinearTiffList)
    LinearTiffList = LinearTiffList[:, ::4, ::4]
    SaveImage(LinearTiffList[0], "dummy1.tif")
    SaveImage(LinearTiffList[1], "dummy2.tif")
    SaveImage(LinearTiffList[2], "dummy3.tif")
    # for i in range(3):
    #     SaveImage(LinearTiffList[i], DefaultOutputFolder + str(i) + ".png")
    WeightsType = ["Tent"]
    for W in WeightsType:
        print("Processing for weighting: ", W)

        OpSaveDir = DefaultOutputFolder + W + "/"
        os.makedirs(OpSaveDir, exist_ok = True)

        HDR1 = MergeLDRImages(LinearTiffList, LinearTiffList, 0.05, 0.95, 0, 65535, W, "Linear")
        #writeEXR(OpSaveDir + "LinearRaw.EXR", HDR1)
        GHDR1 = GamaCorrection(HDR1)
        
        HDR2 = MergeLDRImages(LinearTiffList, LinearTiffList, 0.05, 0.95, 0, 65535, W, "Logarithmic")
        #writeEXR(OpSaveDir + "LogRaw.EXR", HDR2)
        GHDR2 = GamaCorrection(HDR2)
        
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(np.clip(0, 1, GHDR1))
        plt.axis('off')
        plt.title("LinearRaw")
        fig.add_subplot(1, 2, 2)
        plt.imshow(np.clip(0, 1, GHDR2))
        plt.axis('off')
        plt.title("LogRaw")
        plt.savefig(DefaultOutputFolder + W + ".png", bbox_inches='tight', pad_inches=0)
        plt.show()

    HDRDl = ReadRawImage("finalHDR.hdr")
    print(np.mean((HDR2 - HDRDl) ** 2))
    print(time.time() - start)
    
    

    
































