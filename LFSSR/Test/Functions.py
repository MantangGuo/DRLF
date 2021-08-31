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
    indx=random.randrange(0,H-patchSize[2],8)
    indy=random.randrange(0,W-patchSize[3],8)
    # indx=random.randint(0,H-patchSize[2])
    # indy=random.randint(0,W-patchSize[3])
    hrLFPatch=hrLF[:,:,np.newaxis,
                   indx:indx+patchSize[2],
                   indy:indy+patchSize[3]]
                   
    lrLFPatch=lrLF[:,:,np.newaxis,
                   indx//scale_factor:indx//scale_factor+patchSize[2]//scale_factor,
                   indy//scale_factor:indy//scale_factor+patchSize[3]//scale_factor]
    return hrLFPatch,lrLFPatch #[u v c x y] [u v c x/s y/s]
    


def CropLF(lf,patchSize, stride): #lf [b,u,v,x,y,c]
    b,u,v,x,y,c=lf.shape
    numX=len(range(0,x-patchSize,stride))
    numY=len(range(0,y-patchSize,stride))
    lfStack=torch.zeros(b,numX*numY,u,v,patchSize,patchSize,c)

    indCurrent=0
    for i in range(0,x-patchSize,stride):
        for j in range(0,y-patchSize,stride):
            lfPatch=lf[:,:,:,i:i+patchSize,j:j+patchSize,:]
            lfStack[:,indCurrent,:,:,:,:,:]=lfPatch
            indCurrent=indCurrent+1

    return lfStack, [numX,numY] #lfStack [b,n,u,v,x,y,c] 


def MergeLF(lfStack, coordinate, overlap):
    b,n,u,v,x,y,c=lfStack.shape
    
    xMerged=coordinate[0]*x-coordinate[0]*overlap
    yMerged=coordinate[1]*y-coordinate[1]*overlap

    lfMerged=torch.zeros(b,u,v,xMerged,yMerged,c)
    for i in range(coordinate[0]):
        for j in range(coordinate[1]):
            lfMerged[:,
                     :,
                     :,
                     i*(x-overlap):(i+1)*(x-overlap),
                     j*(y-overlap):(j+1)*(y-overlap),
                     :]=lfStack[:,
                                i*coordinate[1]+j,
                                :,
                                :,
                                overlap//2:-overlap//2,
                                overlap//2:-overlap//2,
                                :] 
            
    return lfMerged # [b,u,v,x,y,c]



def ComptPSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
def lfreprojection(lf,phi):
    u,v,x,y,c=lf.shape
    lf=lf/255
    lf2d=np.zeros((x,y,c),dtype=np.float32)
    lf=lf.reshape(u,v,x*y,c)

    for ind_u in range(u):
        for ind_v in range(v):
            offsets=[0]
            phi_uv=sparse.dia_matrix((phi[:,ind_u*u+ind_v],offsets),shape=(x*y,x*y)).toarray()           
            lf2d=lf2d+np.dot(phi_uv,lf[ind_u,ind_v,:,:]).reshape(x,y,c)
    # lf2d=torch.tensor(lf2d,dtype=torch.float32)
    # lf2d=lf2d.permute(2,3,0,1).cuda()
    return lf2d #[x,y,c]