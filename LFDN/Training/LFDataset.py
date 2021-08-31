from __future__ import print_function, division
import os
import scipy.io as scio
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
import h5py
import scipy.io as scio
import numpy as np
import random
import torch.nn.functional as F
from Functions import ExtractPatch,ResizeLF
warnings.filterwarnings("ignore")
plt.ion()

# Loading data
class LFDataset(Dataset):
    """Light Field dataset."""

    def __init__(self,opt):

        super(LFDataset, self).__init__()
        
        dataSet = scio.loadmat(opt.dataPath)
        self.LFSet = dataSet['lf']  #[u, v, x, y, ind]
        self.noiLFSet = dataSet['noilf_{}'.format(opt.sigma)]
        self.patchSize=opt.patchSize


    def __getitem__(self, idx):
        

        LF=self.LFSet[:,:,:,:,idx] #[u, v, x, y]
        noiLF=self.noiLFSet[:,:,:,:,idx] 


        LFPatch,noiLFPatch=ExtractPatch(LF,noiLF,self.patchSize) #[u v c x y]
        
        LFPatch= torch.from_numpy(LFPatch.astype(np.float32)/255)
        noiLFPatch= torch.from_numpy(noiLFPatch.astype(np.float32)/255)
        
        sample = {'LFPatch':LFPatch,'noiLFPatch': noiLFPatch}
        return sample
        
    def __len__(self):
        return self.LFSet.shape[4]



