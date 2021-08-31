import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import scipy.io as scio
from RefNet_Res import RefNet
from Functions import weights_init

warnings.filterwarnings("ignore")
plt.ion()

class StageBlock(torch.nn.Module):
    def __init__(self, kernelSize, opt):
        super(StageBlock,self).__init__()
        
        # Regularization sub-network
        self.refnet=RefNet(opt)
        self.refnet.apply(weights_init)
        
        self.proj=torch.nn.Conv2d(in_channels=opt.angResolution*opt.angResolution,out_channels=opt.measurementNum,kernel_size=1,bias=False)
        torch.nn.init.xavier_uniform_(self.proj.weight.data)
        
        self.recon=torch.nn.ConvTranspose2d(in_channels=opt.channelNum,out_channels=opt.channelNum,kernel_size=kernelSize,bias=False)
        torch.nn.init.xavier_uniform_(self.recon.weight.data)
        
        self.delta=torch.nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda()) 
        self.eta=torch.nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda())
        
    def forward(self,out_lastStage, degLF, originalLFSize):
        b,u,v,c,x,y=originalLFSize
        
        out_refnet = self.refnet(out_lastStage.reshape(b,x,y,c,u,v).permute(0,3,4,5,1,2)) #[b,c,u,v,x,y]
        reProj = self.proj(out_lastStage.reshape(b,x,y,c,u,v).permute(0,3,4,5,1,2).reshape(b*c,u*v,x,y).contiguous())
        err1 = self.recon(reProj.reshape(b,c,1,2,x,y).permute(0,4,5,1,2,3).reshape(b*x*y,c,1,2) - degLF) #[bxy,c,u,v]
        err2 = out_lastStage - out_refnet.permute(0,4,5,1,2,3).reshape(b*x*y,c,u,v) #[bxy,c,u,v]
        out_currentStage = out_lastStage - self.delta * (err1 + self.eta * err2) #[bxy,c,u,v]
        return out_currentStage
        
        
def CascadeStages(block, kernelSize, opt):
    blocks = torch.nn.ModuleList([])
    for _ in range(opt.stageNum):
        blocks.append(block(kernelSize, opt))
    return blocks       
        
        
               
# Main Network construction
class MainNet(torch.nn.Module):
    def __init__(self,opt):
        super(MainNet,self).__init__()
        
        # self.channelNum = opt.channelNum
        if opt.measurementNum == 1:    
            self.kernelSize=[opt.angResolution,opt.angResolution]
        if opt.measurementNum == 2:    
            self.kernelSize=[opt.angResolution,opt.angResolution-1]
        if opt.measurementNum == 4:    
            self.kernelSize=[opt.angResolution-1,opt.angResolution-1]
            
        # Shot layer    
        self.proj_init=torch.nn.Conv2d(in_channels=opt.angResolution*opt.angResolution,out_channels=opt.measurementNum,kernel_size=1,bias=False)
        torch.nn.init.xavier_uniform_(self.proj_init.weight.data)
        
        # Initialize LF from measurements
        self.recon_init=torch.nn.ConvTranspose2d(in_channels=opt.channelNum,out_channels=opt.channelNum,kernel_size=self.kernelSize,bias=False)
        torch.nn.init.xavier_uniform_(self.recon_init.weight.data) 
        
        
        # Iterative stages
        self.iterativeRecon = CascadeStages(StageBlock, self.kernelSize, opt)

        
    def forward(self, lf):
        b,u,v,c,x,y=lf.shape
        degLF=self.proj_init(lf.permute(0,3,1,2,4,5).reshape(b*c,u*v,x,y))#[bc,uDeg*vDeg,x,y]
        degLF = degLF.reshape(b,c,1,2,x,y).permute(0,4,5,1,2,3).reshape(b*x*y,c,1,2) #[bxy,c,1,2]
        initLF=self.recon_init(degLF)#[bxy,c,u,v]
        out=initLF
        
        for stage in self.iterativeRecon:
            out = stage(out, degLF, [b,u,v,c,x,y]) #[bxy,c,u,v]
            
        return out.reshape(b,x,y,c,u,v).permute(0,4,5,3,1,2)#[b,u,v,c,x,y]

