from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
plt.ion()


class RefNet(torch.nn.Module):
    def __init__(self,in_channels):
        super(RefNet,self).__init__()
        self.conv_spa_1=torch.nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_spa_2=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_spa_3=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_spa_4=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_spa_5=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_spa_6=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_spa_7=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_spa_8=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_spa_9=torch.nn.Conv2d(in_channels=64,out_channels=in_channels,kernel_size=3,stride=1,padding=1,bias=True)

        self.conv_ang_1=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_ang_2=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_ang_3=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_ang_4=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_ang_5=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_ang_6=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_ang_7=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_ang_8=torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv_ang_9=torch.nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1,bias=True)
        
        # self.conv_4=torch.nn.Conv3d(in_channels=out_channels,out_channels=out_channels,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1),bias=True)

    def forward(self,input):
        b,c,u,v,x,y=input.shape
        conv_spa_1=F.relu(self.conv_spa_1(input.permute(0,2,3,1,4,5).reshape(b*u*v,c,x,y))) #[64]
        conv_ang_1=F.relu(self.conv_ang_1(conv_spa_1.reshape(b,u,v,64,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,64,u,v))) #[64]

        conv_spa_2=F.relu(self.conv_spa_2(conv_ang_1.reshape(b,x,y,64,u,v).permute(0,4,5,3,1,2).reshape(b*u*v,64,x,y))) #[64]
        conv_ang_2=F.relu(self.conv_ang_2(conv_spa_2.reshape(b,u,v,64,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,64,u,v))+conv_ang_1) #[64]

        conv_spa_3=F.relu(self.conv_spa_3(conv_ang_2.reshape(b,x,y,64,u,v).permute(0,4,5,3,1,2).reshape(b*u*v,64,x,y))) #[64]
        conv_ang_3=F.relu(self.conv_ang_3(conv_spa_3.reshape(b,u,v,64,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,64,u,v))+conv_ang_2) #[64]

        conv_spa_4=F.relu(self.conv_spa_4(conv_ang_3.reshape(b,x,y,64,u,v).permute(0,4,5,3,1,2).reshape(b*u*v,64,x,y))) #[64]
        conv_ang_4=F.relu(self.conv_ang_4(conv_spa_4.reshape(b,u,v,64,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,64,u,v))+conv_ang_3) #[64]

        conv_spa_5=F.relu(self.conv_spa_5(conv_ang_4.reshape(b,x,y,64,u,v).permute(0,4,5,3,1,2).reshape(b*u*v,64,x,y))) #[64]
        conv_ang_5=F.relu(self.conv_ang_5(conv_spa_5.reshape(b,u,v,64,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,64,u,v))+conv_ang_4) #[64]

        conv_spa_6=F.relu(self.conv_spa_6(conv_ang_5.reshape(b,x,y,64,u,v).permute(0,4,5,3,1,2).reshape(b*u*v,64,x,y))) #[3]
        conv_ang_6=F.relu(self.conv_ang_6(conv_spa_6.reshape(b,u,v,64,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,64,u,v))+conv_ang_5) #[3]
        
        conv_spa_7=F.relu(self.conv_spa_7(conv_ang_6.reshape(b,x,y,64,u,v).permute(0,4,5,3,1,2).reshape(b*u*v,64,x,y))) #[3]
        conv_ang_7=F.relu(self.conv_ang_7(conv_spa_7.reshape(b,u,v,64,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,64,u,v))+conv_ang_6) #[3]
        
        conv_spa_8=F.relu(self.conv_spa_8(conv_ang_7.reshape(b,x,y,64,u,v).permute(0,4,5,3,1,2).reshape(b*u*v,64,x,y))) #[3]
        conv_ang_8=F.relu(self.conv_ang_8(conv_spa_8.reshape(b,u,v,64,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,64,u,v))+conv_ang_7) #[3]
        
        conv_spa_9=F.relu(self.conv_spa_9(conv_ang_8.reshape(b,x,y,64,u,v).permute(0,4,5,3,1,2).reshape(b*u*v,64,x,y))) #[3]
        conv_ang_9=self.conv_ang_9(conv_spa_9.reshape(b,u,v,c,x,y).permute(0,4,5,3,1,2).reshape(b*x*y,c,u,v)) #[3]

        output=conv_ang_9.reshape(b,x,y,c,u,v).permute(0,3,4,5,1,2)+input #[3]
        
        return output
