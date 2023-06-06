import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
torch.cuda.set_device(0)
from trainAndTest import *

def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SEAL-CI model.
    """
    losses,accs,testResults = train(trainArgs)
    
if __name__ == "__main__":
    main()