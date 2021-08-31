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

# Generate backward warped center views from other views by disparity map of center view
def esti_centerview(input,disp):
    warped_certers=torch.rand_like(input,dtype=torch.float32,requires_grad=False) #[N,81,h,w]
    for i in range(warped_certers.shape[1]):
            warped_certers[:,i:i+1,:,:]=warp(disp=torch.squeeze(input=disp,dim=1),source=input[:,i:i+1,:,:], source_index=i, target_index=41,an=9)

    return warped_certers

#Backward Warp
def warp(disp, source, source_index, target_index, an):
    # disparity: LF [N,h,w]
    # source:  [N,1,h,w] 
    # index: 0-(an^2-1)
    # an: number of angular dim in one dimention    

    N,_,h,w = source.shape
    source_h = math.floor( source_index / an )
    source_w = source_index % an

    target_h=math.floor( target_index / an )
    target_w = target_index % an

    # generate target grid
    XX = Variable(torch.arange(0,w).view(1,1,w).expand(N,h,w)).type_as(disp) #[N,h,w]
    YY = Variable(torch.arange(0,h).view(1,h,1).expand(N,h,w)).type_as(disp)

    grid_w = XX + disp*(target_w-source_w)
    grid_h = YY + disp*(target_h-source_h)
    
    grid_w_norm = 2.0 * grid_w / (w-1) -1.0
    grid_h_norm = 2.0 * grid_h / (h-1) -1.0
            
    grid = torch.stack((grid_w_norm, grid_h_norm),dim=3) #[N,h,w,2]

    # inverse warp
    target = F.grid_sample(source,grid) # [N,1,h,w] warped source
    return target


def rgb2yCbCr(input_im):
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.tensor([[0.257, -0.148, 0.439],
                        [0.564, -0.291, -0.368],
                        [0.098, 0.439, -0.071]])
    bias = torch.tensor([16.0/255.0, 128.0/255.0, 128.0/255.0])
    temp = im_flat.mm(mat) + bias
    out = temp.view(input_im.shape[0], input_im.shape[1],input_im.shape[2])
    return out


def epi2fft(lf,downsample):
    if downsample:
        lf=F.interpolate(lf,scale_factor=(1,0.5,0.5))
        N,u,v,h,w=lf.shape
        epi_fft_horizontal=torch.rand((N,h,v,w,2),dtype=torch.float32,requires_grad=False) # [N,h,v,w,2]
        epi_fft_vertical=torch.rand((N,w,u,h,2),dtype=torch.float32,requires_grad=False) # [N,w,u,h,2]
        # horizontal
        for i in range(h):
            epi_fft_horizontal[:,i,:,:,:]=torch.rfft(lf[:,3,:,i,:],2,onesided=False)
        # vertical
        for i in range(w):
            epi_fft_vertical[:,i,:,:,:]=torch.rfft(lf[:,:,3,:,i],2,onesided=False)
        # # diagram
        # for i in range(h):
        #     for j in range(u):  
        #         epi_fft_diagram[:,i,:,:,:]=torch.rfft(lf[:,:,3,:,i],2,onesided=False)
        # # inverse diagram
        # for i in u
        #     epi_fft_indiagram[:,i:i+1,h,w,2]=torch.rfft(F.interpolate(lf[:,end-i-1:end-i,i:i+1,:,:],scale_factor=0.5),2,onesided=False))
    else:
        N,u,v,h,w=lf.shape
        epi_fft_horizontal=torch.rand((N,h,v,w,2),dtype=torch.float32,requires_grad=False) # [N,h,v,w,2]
        epi_fft_vertical=torch.rand((N,w,u,h,2),dtype=torch.float32,requires_grad=False) # [N,w,u,h,2]
        # horizontal
        for i in range(h):
            epi_fft_horizontal[:,i,:,:,:]=torch.rfft(lf[:,3,:,i,:],2,onesided=False)
        # vertical
        for i in range(w):
            epi_fft_vertical[:,i,:,:,:]=torch.rfft(lf[:,:,3,:,i],2,onesided=False)
        # # diagram
        # for i in u
        #     epi_fft_diagram[:,i:i+1,h,w,2]=torch.rfft(lf[:,i:i+1,i:i+1,:,:],2,onesided=False))
        # # inverse diagram
        # for i in u
        #     epi_fft_indiagram[:,i:i+1,h,w,2]=torch.rfft(lf[:,end-i-1:end-i,i:i+1,:,:],2,onesided=False))

    #2-Norm and log 
    epi_fft_horizontal=torch.log(torch.norm(epi_fft_horizontal,p=2,dim=4)+1)
    epi_fft_vertical=torch.log(torch.norm(epi_fft_vertical,p=2,dim=4)+1)

    return epi_fft_horizontal, epi_fft_vertical


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

            

def matrix_compress(nonzero_data,block_num,block_h,block_w):
    offsets=[0]
    matrix=sparse.dia_matrix((nonzero_data[:,0],offsets),shape=(block_h*block_w,block_h*block_w))
    for i in range(block_num-1):
        block=sparse.dia_matrix((nonzero_data[:,i+1],offsets),shape=(block_h*block_w,block_h*block_w))
        matrix=sparse.hstack([matrix,block])
    return matrix

# def matrix_compress(nonzero_data,block_num,block_h,block_w):
#     index=torch.LongTensor([range(200*200),range(200*200)])
#     value=torch.FloatTensor(nonzero_data[:,0])
#     matrix=torch.sparse.FloatTensor(index,value)
#     for i in range(block_num-1):
#         v=torch.FloatTensor(nonzero_data[:,i+1])
#         block=torch.sparse.FloatTensor(index,value)
#         matrix=torch.sparse.FloatTensor.coalesce(matrix,block)
#     return matrix


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
    
def ResizeLF(lf,scale_factor):
    u,v,x,y,c=lf.shape
    resizedLF=np.zeros((u,v,int(scale_factor*x),int(scale_factor*y),c),dtype=np.int)
    for ind_u in range(u):
        for ind_v in range(v):
            view=lf[ind_u,ind_v,:,:,:]
            resizedView=cv2.resize(view, (int(scale_factor*x),int(scale_factor*y)), interpolation=cv2.INTER_CUBIC)
            resizedLF[ind_u,ind_v,:,:,:]=resizedView.reshape(int(scale_factor*x),int(scale_factor*y),-1)
    return resizedLF

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