from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
import h5py
import numpy as np
from Functions import ExtractPatch
warnings.filterwarnings("ignore")
plt.ion()

# Loading data
class LFDataset(Dataset):
    """Light Field dataset."""

    def __init__(self, opt):

        super(LFDataset, self).__init__()
        
        dataSet = h5py.File(opt.dataPath)
        self.hrLFset = dataSet.get('img_HR')[:]  #[ind, u, v, x, y]
        self.lrLFset = dataSet.get('img_LR_{}'.format(opt.scaleFactor))[:]  #[ind, u, v, x/s, y/s]
        self.hrSize = dataSet.get('img_size')[:] #[ind, H,W] The spatial resolution of hrLF
        self.patchSize=opt.patchSize
        self.scaleFactor=opt.scaleFactor



    def __getitem__(self, idx):
        

        hrLF=self.hrLFset[idx] #[u, v, x, y]
        lrLF=self.lrLFset[idx] #[u, v, x/s, y/s]
        H,W=self.hrSize[idx] #[H,W]

        hrLFPatch,lrLFPatch=ExtractPatch(hrLF,lrLF, H, W, self.patchSize, self.scaleFactor) #[u v c x y] [u v c x/s y/s]
        hrLFPatch= torch.from_numpy(hrLFPatch.astype(np.float32)/255)
        lrLFPatch= torch.from_numpy(lrLFPatch.astype(np.float32)/255)
        sample = {'hrLF':hrLFPatch,'lrLF': lrLFPatch}
        return sample
        
    def __len__(self):
        return self.hrLFset.shape[0]



