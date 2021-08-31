from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
# import h5py
import scipy.io as scio
import numpy as np
from Functions import ExtractPatch
warnings.filterwarnings("ignore")
plt.ion()


class LFDataset(Dataset):

# root_dir: the path of .mat file.
# scale_factor: the upsamle scale factor

    def __init__(self, opt):
        super(LFDataset, self).__init__()   
        dataSet = scio.loadmat(opt.dataPath)
        self.hrLF_yset = dataSet['GT_y']  #[ind, u, v, x, y]
        self.lrLFset = dataSet['LR_ycbcr']  #[ind, u, v, c, x/s, y/s]
        self.nameLFset = dataSet['LF_name'] #[ind, name] Each testing LF's name is saved as a series of ASCIIs
        self.scaleFactor=opt.scaleFactor


    def __getitem__(self, idx):
        
        H=self.hrLF_yset.shape[3]
        W=self.hrLF_yset.shape[4]
        
        hrLF_y=self.hrLF_yset[idx][:,:,np.newaxis,:,:] #[u, v, c, x, y]
        lrLF=self.lrLFset[idx]#[u, v, c, x/s, y/s]
        nameLF=''.join([chr(self.nameLFset[idx][0][0][i]) for i in range(self.nameLFset[idx][0][0].shape[0])])

        hrLF_y= torch.from_numpy(hrLF_y.astype(np.float32)/255)
        lrLF= torch.from_numpy(lrLF.astype(np.float32)/255)
        
        hrLF_bicubic=torch.nn.functional.interpolate(lrLF.reshape(lrLF.shape[0]*lrLF.shape[1],lrLF.shape[2],lrLF.shape[3],lrLF.shape[4],),scale_factor=self.scaleFactor,mode='bicubic',align_corners=False).reshape(lrLF.shape[0],lrLF.shape[1],lrLF.shape[2],H,W) #[u, v, c, x, y]
        
        sample = {'hrLF_y':hrLF_y.permute(0,1,3,4,2),
                  'lrLF_y': lrLF[:,:,0:1,:,:].permute(0,1,3,4,2), 
                  'hrLF_bicubic': hrLF_bicubic.permute(0,1,3,4,2),
                  'nameLF':nameLF}
                  
        return sample
        
    def __len__(self):
        return self.hrLF_yset.shape[0]



