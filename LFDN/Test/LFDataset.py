from __future__ import print_function, division
import os
import scipy.io as scio
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
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

    def __init__(self, opt):

        super(LFDataset, self).__init__()
        
        dataSet = scio.loadmat(opt.dataPath)

        self.LFSet = dataSet['lf']  #[u, v, x, y,ind]
        self.noiLFSet = dataSet['noilf_{}'.format(opt.sigma)] #[u, v, x, y,ind]
        self.lfNameSet = dataSet['LF_name'] #[index, 1]
    def __getitem__(self, idx):
        
        LF=self.LFSet[:,:,:,:,idx] #[u, v, x, y]
        noiLF=self.noiLFSet[:,:,:,:,idx] 
        lfName=''.join([chr(self.lfNameSet[idx][0][0][i]) for i in range(self.lfNameSet[idx][0][0].shape[0])])

        
        LF= torch.from_numpy(LF[:,:,:,:,np.newaxis].astype(np.float32)/255.0)
        noiLF= torch.from_numpy(noiLF[:,:,:,:,np.newaxis].astype(np.float32)/255.0) #[u,v,x,y,c]
        
        sample = {'LF':LF,'noiLF': noiLF,'lfName':lfName}
        return sample
        
    def __len__(self):
        return self.LFSet.shape[4]



