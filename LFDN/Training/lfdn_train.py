import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
from LFDataset import LFDataset
from Functions import weights_init,SetupSeed
from DeviceParameters import to_device
from MainNet import MainNet
import itertools,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('Training_SASOnly_Res_%s.log' % datetime.now().strftime("%Y%m%d%H%M"))
log.addHandler(fh)


# Training settings
parser = argparse.ArgumentParser(description="Light Field Denoising")
parser.add_argument("--learningRate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--sigma", type=int, default=10, help="Noise level")
parser.add_argument("--stageNum", type=int, default=3, help="The number of stages")
parser.add_argument("--batchSize", type=int, default=3, help="Batch size")
parser.add_argument("--sampleNum", type=int, default=70, help="The number of LF in training set")
parser.add_argument("--patchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--channelNum", type=int, default=1, help="The number of input channels")
parser.add_argument("--epochNum", type=int, default=10000, help="The number of epoches")
parser.add_argument("--summaryPath", type=str, default='./', help="Path for saving training log ")
parser.add_argument("--dataPath", type=str, default='/home/guo_19/data/train_noiseLevel_10-20-50_4-11.mat', help="Path for loading training data ")

opt = parser.parse_args()
logging.info(opt)

if __name__ == '__main__':

    SetupSeed(1)
    savePath = './model/lfdn_SASOnly_Res{}.pth'.format(opt.sigma)
    lfDataset = LFDataset(opt)
    dataloader = DataLoader(lfDataset, batch_size=opt.batchSize,shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


    model=MainNet(opt)
    # model.load_state_dict(torch.load('./model/lfdn_SASOnly_Res20.pth'))
    # model.eval()
    #model.apply(weights_init)
    to_device(model,device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Training parameters: %d" %total_trainable_params)

    criterion = torch.nn.L1Loss() # Loss 

    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=opt.learningRate) #optimizer
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.epochNum*0.8, gamma=0.1, last_epoch=-1)
    writer = SummaryWriter(opt.summaryPath)
    

    lossLogger = defaultdict(list)

    for epoch in range(opt.epochNum):
        batch = 0
        lossSum = 0
        for _,sample in enumerate(dataloader):
            batch = batch +1
            LF=sample['LFPatch']
            noiLF=sample['noiLFPatch']
            LF = to_device(LF,device) # label:[u v c x y] 
            noiLF= to_device(noiLF,device)  # input:[u v c x/s y/s]
            
            estimatedLF=model(noiLF)
            loss = criterion(estimatedLF,LF)
            lossSum += loss.item()            
            
            writer.add_scalar('loss', loss, opt.sampleNum//opt.batchSize*epoch+batch)
            print("Epoch: %d Batch: %d Loss: %.6f" %(epoch,batch,loss.item()))
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(),savePath)
        log.info("Epoch: %d Loss: %.6f" %(epoch,lossSum/len(dataloader)))
        scheduler.step()
        
        #Record the training loss
        lossLogger['Epoch'].append(epoch)
        lossLogger['Loss'].append(lossSum/len(dataloader))
        plt.figure()
        plt.title('Loss')
        plt.plot(lossLogger['Epoch'],lossLogger['Loss'])
        plt.savefig('Training_{}.jpg'.format(opt.sigma))
        plt.close()
    

