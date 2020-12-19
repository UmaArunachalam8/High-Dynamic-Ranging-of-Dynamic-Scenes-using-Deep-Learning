import numpy as np 
import torch 
import math
import argparse
import skimage.io
#import OpenEXR
TrainAlignedImagesLDRDir = "../data/AlignedImagesTrain/"
TrainGtResultsRootDir = "../data/Trainingset/Training/"
TestAlignedImagesLDRDir = "../data/AlignmentImagesTest/"
TestGtResultsRootDir = "../data/Testset/Test/"
DefaultModelStorePath = "../models/Model.pth"

# TrainAlignedImagesLDRDir = "/mnt/data/data/AlignedImagesTrain/"
# TrainGtResultsRootDir = "/mnt/data/data/Trainingset/Training/"
# TestAlignedImagesLDRDir = "/mnt/data/data/AlignmentImagesTest/"
# TestGtResultsRootDir = "/mnt/data/data/Testset/Test/"
# DefaultModelStorePath = "../models/Model.pth"

def ParseArgs():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    
    parser = argparse.ArgumentParser(description='GeoViz Project')
    parser.add_argument('--Model', type=str, choices=['SimpleFlowBaseline', 'SimpleDepthBaseline', 'WithDepthBaseline', 'WithPoseBaseline'], default='WithPoseBaseline')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables/disables CUDA training')

    parser.add_argument('--TrainDataDir', default=TrainAlignedImagesLDRDir, help='dataset for training')
    parser.add_argument('--TrainLabelDir', default=TrainGtResultsRootDir, help='labels for training')

    parser.add_argument('--ValDataDir', default=TestAlignedImagesLDRDir, help='dataset for val')
    parser.add_argument('--ValLabelDir', default=TestGtResultsRootDir, help='labels for val')

    parser.add_argument('--ModelStorePath', default=DefaultModelStorePath, help='save trained model')

    parser.add_argument('--NetworkModel', default='WE', help='type of model used for experiments') #WE vs Direct

    parser.add_argument('--NumEpochs', type=int, default=1000, metavar='N', help='number of epochs to train')
    parser.add_argument('--BatchSize', type=int, default=1, metavar='N', help='size of training/val batch')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')

    Args = parser.parse_args()
    UseCuda = not Args.no_cuda and torch.cuda.is_available()
    Device = torch.device("cuda" if UseCuda else "cpu")
    return Args, Device


def ReadRawImage(InPath):
    Image = skimage.io.imread(InPath)
    Image = Image.astype(np.double)
    return Image

def SaveImage(Output, Path):
    Output = np.clip(0, 1, Output)
    skimage.io.imsave(Path, Output)

def Normalize(InImage, InMin, InMax):
    # rescale to range [0, 1] from [Black, White]
    InImage = (InImage - InMin) / (InMax - InMin)
    # clip outliers
    # InImage = np.where(InImage > 1.0, 1, InImage)
    # InImage = np.where(InImage < 0.0, 0, InImage)
    return InImage

def GamaCorrection(InImage):
    InImage = np.where(InImage <= 0.0031308, 12.92 * InImage, ((1 + 0.055) * InImage ** (1 / 2.4)) - 0.055)
    return InImage

# def writeEXR(name, data):
#     """ Write EXR file from data (dtype=np.float)
#     """
#     exr = OpenEXR.OutputFile(name, OpenEXR.Header(data.shape[1], data.shape[0]))

#     R = (data[:,:,0]).astype(np.float32).tobytes()
#     G = (data[:,:,1]).astype(np.float32).tobytes()
#     B = (data[:,:,2]).astype(np.float32).tobytes()

#     exr.writePixels({'R' : R, 'G' : G, 'B' : B })
#     exr.close()

def ToneMapImage(H):
    Mu = 5000
    Image = np.log(1 + Mu * H) / np.log(1 + Mu)
    return Image

def LDRToHDR(Images, Exposures):
    Images = Images ** 2.2 / Exposures.reshape(-1, 1, 1, 1)
    return Images

def save_checkpoint(checkpoint_path, model, optimizer):
    # this function is taken from CS231n pytorch tutorial
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    # this function is taken from CS231n pytorch tutorial
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    return model, optimizer

def EstimatePSNR(Prediction, Label):
    MSE = torch.mean((Prediction - Label) ** 2)
    return 10 * math.log10(1/MSE)

def writeEXR(name, data):
    """ Write EXR file from data (dtype=np.float)
    """
    exr = OpenEXR.OutputFile(name, OpenEXR.Header(data.shape[1], data.shape[0]))

    R = (data[:,:,0]).astype(np.float32).tobytes()
    G = (data[:,:,1]).astype(np.float32).tobytes()
    B = (data[:,:,2]).astype(np.float32).tobytes()

    exr.writePixels({'R' : R, 'G' : G, 'B' : B })
    exr.close()


