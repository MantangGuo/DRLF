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
    def __init__(self,opt):
        super(StageBlock,self).__init__()
        self.scaleFactor=opt.scaleFactor
        self.refnet=RefNet(opt)
        self.refnet.apply(weights_init)
        
        self.proj=torch.nn.Conv2d(in_channels=opt.channelNum,out_channels=opt.channelNum,kernel_size=self.scaleFactor*2,stride=self.scaleFactor, padding=int(self.scaleFactor/2),groups=opt.channelNum,bias=False)
        torch.nn.init.xavier_uniform_(self.proj.weight.data)
        
        self.recon=torch.nn.ConvTranspose2d(in_channels=opt.channelNum,out_channels=opt.channelNum,kernel_size=self.scaleFactor*2,stride=self.scaleFactor, padding=int(self.scaleFactor/2),groups=opt.channelNum,bias=False)
        torch.nn.init.xavier_uniform_(self.recon.weight.data)
        
        self.delta=torch.nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda()) 
        self.eta=torch.nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda())
        
    def forward(self,out_lastStage, degLF):
        b,u,v,c,x,y=degLF.shape
        hx=int(x*self.scaleFactor)
        hy=int(y*self.scaleFactor)
        
        out_refnet = self.refnet(out_lastStage.reshape(b,u,v,c,hx,hy).permute(0,3,1,2,4,5)) #[b,c,u,v,x,y]
        err1 = self.recon(self.proj(out_lastStage) - degLF.reshape(b*u*v,c,x,y)) #[bxy,c,u,v]
        err2 = out_lastStage - out_refnet.permute(0,2,3,1,4,5).reshape(b*u*v,c,hx,hy) #[bxy,c,u,v]
        out_currentStage = out_lastStage - self.delta * (err1 + self.eta * err2) #[bxy,c,u,v]
        return out_currentStage
        
        
def CascadeStages(block, opt):
    blocks = torch.nn.ModuleList([])
    for _ in range(opt.stageNum):
        blocks.append(block(opt))
    return blocks       
        
        
               
# Main Network construction
class MainNet(torch.nn.Module):
    def __init__(self,opt):
        super(MainNet,self).__init__()
        self.scaleFactor=opt.scaleFactor
        
        self.iterativeRecon = CascadeStages(StageBlock,opt)
        self.recon_init=torch.nn.ConvTranspose2d(in_channels=opt.channelNum,out_channels=opt.channelNum,kernel_size=self.scaleFactor*2,stride=self.scaleFactor, padding=int(self.scaleFactor/2),groups=opt.channelNum,bias=False)
        torch.nn.init.xavier_uniform_(self.recon_init.weight.data)
        
    def forward(self, degLF):
        b,u,v,c,x,y=degLF.shape
        hx=int(x*self.scaleFactor)
        hy=int(y*self.scaleFactor)
        
        initLF=self.recon_init(degLF.reshape(b*u*v,c,x,y))#[buv,c,hx,hy]
        out=initLF
        for stage in self.iterativeRecon:
            out = stage(out, degLF)
        return out.reshape(b,u,v,c,hx,hy)

