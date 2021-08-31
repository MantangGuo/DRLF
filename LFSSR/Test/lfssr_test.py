from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils
import warnings
from LFDataset import LFDataset
from DeviceParameters import to_device
from MainNet import MainNet
from Functions import CropLF, MergeLF,ComptPSNR
from skimage.measure import compare_ssim 
import numpy as np
import scipy.io as scio 
import scipy.misc as scim
import os
import logging,argparse
from datetime import datetime

warnings.filterwarnings("ignore")
plt.ion()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('Testing_TIP_2x=.log')
log.addHandler(fh)


# Testing settings
parser = argparse.ArgumentParser(description="Light Field Compressed Sensing")
parser.add_argument("--scaleFactor", type=int, default=2, help="Scale factor for super-resolution")
parser.add_argument("--stageNum", type=int, default=4, help="The number of stages")
parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
# parser.add_argument("--sasNum", type=int, default=10, help="The number of SAS layers")
# parser.add_argument("--angResolution", type=int, default=8, help="The angular resolution of original LF")
parser.add_argument("--cropPatchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--overlap", type=int, default=4, help="The size of croped LF patch")
# parser.add_argument("--kernelSize", type=int, default=4, help="The size of croped LF patch")
parser.add_argument("--channelNum", type=int, default=1, help="The number of input channels")
parser.add_argument("--dataPath", type=str, default='/public/mantanguo/Dataset/LFSSR/test_Stanford_General_x2_TIP.mat', help="Path for loading testing data ")

opt = parser.parse_args()
logging.info(opt)

if __name__ == '__main__':

    cropStride=opt.cropPatchSize-opt.overlap


    load_path='./model/lfssr_4Stages_2x_lyt.pth' #model
    save_path='./testResults/' # test result 


    lf_dataset = LFDataset(opt)

    dataloader = DataLoader(lf_dataset, batch_size=opt.batchSize,shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model=MainNet(opt)
    model.load_state_dict(torch.load(load_path))
    model.eval()
    to_device(model,device)


    with torch.no_grad():
        num = 0
        avg_psnr_y = 0
        avg_ssim_y = 0
        for _,sample in enumerate(dataloader):
            num=num+1
            hrLF_y=sample['hrLF_y'] #test lf
            lrLF_y=sample['lrLF_y'] #test lf
            hrLF_ycbcr=sample['hrLF_bicubic'] #test lf
            nameLF=sample['nameLF'] # the name of test lf

                        
            lrLF_yStack,coordinate=CropLF(lrLF_y,opt.cropPatchSize//opt.scaleFactor, cropStride//opt.scaleFactor) #[b,n,u,v,x,y,c] 
            b,n,u,v,x,y,c=lrLF_yStack.shape
            hrLF_yStack=torch.zeros(b,n,u,v,
                                   int(x*opt.scaleFactor),
                                   int(y*opt.scaleFactor),
                                   c)#[b,n,u,v,x,y,c]
                                   

                        
            # reconstruction
            for i in range(n):
                hrLF_yPatch=model(lrLF_yStack[:,i,:,:,:,:].permute(0,1,2,5,3,4).cuda())  #[b,u,v,c,x,y]
                hrLF_yStack[:,i,:,:,:,:,:]=hrLF_yPatch.permute(0,1,2,4,5,3) #[b,n,u,v,x,y,c]
                
  
                         
            hrLF_y_esti=MergeLF(hrLF_yStack,coordinate,opt.overlap) # the overlap between adjacent patch will be larger along with scaleFactor increasing
            
            b,u,v,x,y,c=hrLF_y_esti.shape
            hrLF_ycbcr=hrLF_ycbcr[:,:,:,
                                  opt.overlap//2:opt.overlap//2+x,
                                  opt.overlap//2:opt.overlap//2+y,:]
                                  
            hrLF_y=hrLF_y[:,:,:,
                          opt.overlap//2:opt.overlap//2+x,
                          opt.overlap//2:opt.overlap//2+y,:]
                          
            hrLF_ycbcr[:,:,:,:,:,0:1]=hrLF_y_esti 
            
            
            lf_psnr_y = 0
            lf_ssim_y = 0
   
            
            for ind_uv in range(u*v):

                lf_psnr_y += ComptPSNR(np.squeeze(hrLF_y.reshape(b,u*v,x,y,c)[0,ind_uv].cpu().numpy()),
                                       np.squeeze(hrLF_y_esti.reshape(b,u*v,x,y,c)[0,ind_uv].cpu().numpy()))  / (u*v)
                                       
                lf_ssim_y += compare_ssim(np.squeeze((hrLF_y.reshape(b,u*v,x,y,c)[0,ind_uv].cpu().numpy()*255.0).astype(np.uint8)),
                                          np.squeeze((hrLF_y_esti.reshape(b,u*v,x,y,c)[0,ind_uv].cpu().numpy()*255.0).astype(np.uint8)),gaussian_weights=True,sigma=1.5,use_sample_covariance=False) / (u*v)

            avg_psnr_y += lf_psnr_y / len(dataloader)           
            avg_ssim_y += lf_ssim_y / len(dataloader) 
            
            
            # save reconstructed LF
            # scio.savemat(os.path.join(save_path,nameLF[0]+'.mat'),
            #              {'lf_recons':torch.squeeze(hrLF_ycbcr).numpy()}) #[u,v,x,y,c]

            
            log.info('Index: %d  Scene: %s  PSNR: %.2f  SSIM: %.3f'%(num,nameLF[0],lf_psnr_y,lf_ssim_y))
        log.info('Average PSNR: %.2f  SSIM: %.3f '%(avg_psnr_y,avg_ssim_y))            