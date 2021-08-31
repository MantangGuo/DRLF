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
    def __init__(self,channelNum):
        super(StageBlock,self).__init__()
        self.refnet=RefNet(channelNum)
        self.refnet.apply(weights_init)
        
        self.delta=torch.nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda()) 
        self.eta=torch.nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda())
        
    def forward(self,out_lastStage, degLF):
        b,u,v,c,x,y=degLF.shape
        
        out_refnet = self.refnet(out_lastStage.reshape(b,u,v,c,x,y).permute(0,3,1,2,4,5)) #[b,c,u,v,x,y]
        err1 = out_lastStage - degLF.reshape(b*u*v,c,x,y) #[bxy,c,u,v]
        err2 = out_lastStage - out_refnet.permute(0,2,3,1,4,5).reshape(b*u*v,c,x,y) #[bxy,c,u,v]
        out_currentStage = out_lastStage - self.delta * (err1 + self.eta * err2) #[bxy,c,u,v]
        return out_currentStage
        
        
def CascadeStages(block, stageNum,channelNum):
    blocks = torch.nn.ModuleList([])
    for _ in range(stageNum):
        blocks.append(block(channelNum))
    return blocks       
        
        
               
# Main Network construction
class MainNet(torch.nn.Module):
    def __init__(self,opt):
        super(MainNet,self).__init__()

        self.stageNum=opt.stageNum
        self.channelNum=opt.channelNum
        self.iterativeRecon = CascadeStages(StageBlock,self.stageNum,self.channelNum)

        
    def forward(self, degLF):
        b,u,v,c,x,y=degLF.shape

        
        initLF=degLF.reshape(b*u*v,c,x,y)
        out=initLF
        
        for stage in self.iterativeRecon:
            out = stage(out, degLF)
            
        return out.reshape(b,u,v,c,x,y)

