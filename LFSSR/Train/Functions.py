from __future__ import print_function, division
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
# Ignore warnings
import cv2
import warnings
from scipy import sparse
import random
import numpy as np
warnings.filterwarnings("ignore")
plt.ion()



#Initiate parameters in model 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Conv3d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('ConvTranspose3d') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def ExtractPatch(hrLF, lrLF, H, W, patchSize, scale_factor):
    indx=random.randrange(0,H-patchSize,8)
    indy=random.randrange(0,W-patchSize,8)
    # indx=random.randint(0,H-patchSize[2])
    # indy=random.randint(0,W-patchSize[3])
    hrLFPatch=hrLF[:,:,np.newaxis,
                   indx:indx+patchSize,
                   indy:indy+patchSize]
                   
    lrLFPatch=lrLF[:,:,np.newaxis,
                   indx//scale_factor:indx//scale_factor+patchSize//scale_factor,
                   indy//scale_factor:indy//scale_factor+patchSize//scale_factor]
    return hrLFPatch,lrLFPatch #[u v c x y] [u v c x/s y/s]
    

