import utils
import argparse
from RunnerDeepHDR import * 

UsePrevModel = False

if __name__ == "__main__":
    DefaultModelDir = '../models/'
    os.makedirs(DefaultModelDir, exist_ok = True)
    
    Args, Device = utils.ParseArgs()

    ExperimentRunner = RunnerDeepHDR(Args.TrainDataDir, Args.TrainLabelDir, Args.ValDataDir, Args.ValLabelDir, 
        Args.NetworkModel, Args.BatchSize, Args.NumEpochs, Args.lr, Device, Args.ModelStorePath, UsePrevModel)
    ExperimentRunner.Train()
