from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils
import warnings
from LFDataset import LFDataset
from DeviceParameters import to_device
from MainNet import MainNet
from Functions import CropLF, MergeLF,ComptPSNR,rgb2ycbcr
from skimage.measure import compare_ssim 
import numpy as np
import scipy.io as scio 
import scipy.misc as scim
import os,time
import logging,argparse
from datetime import datetime

warnings.filterwarnings("ignore")
plt.ion()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('Testing_syn_sparse_m2.log')
log.addHandler(fh) 

# Testing settings
parser = argparse.ArgumentParser(description="Light Field Denoising")
parser.add_argument("--stageNum", type=int, default=3, help="The number of stages")
# parser.add_argument("--sasNum", type=int, default=9, help="The number of stages")
parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
parser.add_argument("--cropPatchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--overlap", type=int, default=4, help="The size of croped LF patch")
parser.add_argument("--measurementNum", type=int, default=2, help="The number of measurements")
parser.add_argument("--angResolution", type=int, default=7, help="The angular resolution of original LF")
parser.add_argument("--channelNum", type=int, default=1, help="The number of input channels")
parser.add_argument("--modelPath", type=str, default='./model/lyt_m2_3stage_6e-05_1-5.pth', help="Path for loading trained model ")
parser.add_argument("--dataPath", type=str, default='../Dataset/LFCA/test_LFCA_Kalantari_4-10.mat', help="Path for loading testing data ")
parser.add_argument("--savePath", type=str, default='./testResults/', help="Path for saving results ")

opt = parser.parse_args()
logging.info(opt)

if __name__ == '__main__':

    lf_dataset = LFDataset(opt)
    dataloader = DataLoader(lf_dataset, batch_size=opt.batchSize,shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model=MainNet(opt)
    model.load_state_dict(torch.load(opt.modelPath))
    model.eval()
    to_device(model,device)
    

    with torch.no_grad():
        num = 0
        avg_psnr = 0
        avg_ssim = 0
        for _,sample in enumerate(dataloader):
            num=num+1
            LF=sample['LF'] #test lf [b,u,v,x,y,c]
            lfName=sample['lfName']
            b,u,v,x,y,c = LF.shape             
            
            # Crop the input LF into patches 
            LFStack,coordinate=CropLF(LF.permute(0,1,2,5,3,4),opt.cropPatchSize, opt.overlap) #[b,n,u,v,c,x,y]
            n=LFStack.shape[1]
            estiLFStack=torch.zeros(b,n,u,v,c,opt.cropPatchSize,opt.cropPatchSize)#[b,n,u,v,c,x,y]
                                 

            #########################################################
            start = time.time()
            print('start to calculate')
            #########################################################            
            # reconstruction
            for i in range(LFStack.shape[1]):
                estiLFPatch=torch.zeros(b,u,v,c,opt.cropPatchSize,opt.cropPatchSize)#[b,u,v,c,x,y]
                for j in range(c):
                    estiLFPatch[:,:,:,j:j+1,:,:]=model(LFStack[:,i,:,:,j:j+1,:,:].cuda())  #[b,measurementNum,c,x,y] [b,u,v,c,x,y].
                estiLFStack[:,i,:,:,:,:,:]=estiLFPatch #[b,n,u,v,c,x,y]
            
            #########################################################
            end = time.time()
            print('use:',end-start)
            #########################################################    

            # Merge the patches into LF
            estiLF=MergeLF(estiLFStack,coordinate,opt.overlap,x,y) #[b,u,v,c,x,y]
            
            b,u,v,c,xCrop,yCrop=estiLF.shape
            LF=LF[:,:,:, opt.overlap//2:opt.overlap//2+xCrop,opt.overlap//2:opt.overlap//2+yCrop,:]
                                   

            lf_psnr = 0
            lf_ssim = 0

            #RGB channel
            # for ind_uv in range(u*v):

            #     lf_psnr += ComptPSNR(np.squeeze(estiLF.permute(0,1,2,4,5,3).reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy()),
            #                          np.squeeze(LF.reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy()))  / (u*v)
                                       
            #     lf_ssim += compare_ssim(np.squeeze(estiLF.permute(0,1,2,4,5,3).reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy()*255.0).astype(np.uint8),
            #                             np.squeeze(LF.reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy()*255.0).astype(np.uint8),gaussian_weights=True,sigma=1.5,use_sample_covariance=False,multichannel=True) / (u*v)

            # Y channel
            for ind_uv in range(u*v):
                    lf_psnr += ComptPSNR(rgb2ycbcr(estiLF.permute(0,1,2,4,5,3).reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy())[:,:,0],
                                         rgb2ycbcr(LF.reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy())[:,:,0])  / (u*v)
                                        
                    lf_ssim += compare_ssim(rgb2ycbcr(estiLF.permute(0,1,2,4,5,3).reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy()*255.0)[:,:,0].astype(np.uint8),
                                            rgb2ycbcr(LF.reshape(b,u*v,xCrop,yCrop,c)[0,ind_uv].cpu().numpy()*255.0)[:,:,0].astype(np.uint8),gaussian_weights=True,sigma=1.5,use_sample_covariance=False,multichannel=False) / (u*v)
            
            avg_psnr += lf_psnr / len(dataloader)           
            avg_ssim += lf_ssim / len(dataloader)
            log.info('Index: %d  Scene: %s  PSNR: %.2f  SSIM: %.3f'%(num,lfName[0],lf_psnr,lf_ssim))
            

            #save reconstructed LF
            scio.savemat(os.path.join(opt.savePath,lfName[0]+'.mat'),
                         {'lf_recons':torch.squeeze(estiLF).numpy()}) #[u,v,x,y,c]

        log.info('Average PSNR: %.2f  SSIM: %.3f '%(avg_psnr,avg_ssim))              

        # #save coded mask
        # if opt.measurementNum==1:
        #     scio.savemat(os.path.join(opt.savePath, 'mask.mat'),
        #                      {'mask': torch.squeeze(model._modules['proj_init'].weight.data.reshape(-1,opt.angResolution,opt.angResolution).permute(1,2,0)).cpu().numpy()})
        #     scim.imsave(os.path.join(opt.savePath, 'mask.png'),
        #                      torch.squeeze(255.0*model._modules['proj_init'].weight.data.reshape(-1,opt.angResolution,opt.angResolution).permute(1,2,0)).cpu().numpy())
        # if opt.measurementNum==2:
        #     scio.savemat(os.path.join(opt.savePath, 'mask.mat'),
        #                      {'mask': torch.squeeze(model._modules['proj_init'].weight.data.reshape(-1,opt.angResolution,opt.angResolution-1).permute(1,2,0)).cpu().numpy()})
        #     scim.imsave(os.path.join(opt.savePath, 'mask.png'),
        #                      torch.squeeze(255.0*model._modules['proj_init'].weight.data.reshape(-1,opt.angResolution,opt.angResolution-1).permute(1,2,0)).cpu().numpy())
        # if opt.measurementNum==4:
        #     scio.savemat(os.path.join(opt.savePath, 'mask.mat'),
        #                      {'mask': torch.squeeze(model._modules['proj_init'].weight.data.reshape(-1,opt.angResolution-1,opt.angResolution-1).permute(1,2,0)).cpu().numpy()})
        #     scim.imsave(os.path.join(opt.savePath, 'mask.png'),
        #                      torch.squeeze(255.0*model._modules['proj_init'].weight.data.reshape(-1,opt.angResolution-1,opt.angResolution-1).permute(1,2,0)).cpu().numpy())

        
                   