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




# Training settings
parser = argparse.ArgumentParser(description="Light Field Compressed Sensing")
parser.add_argument("--learningRate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--stageNum", type=int, default=6, help="The number of stages")
parser.add_argument("--batchSize", type=int, default=5, help="Batch size")
parser.add_argument("--sampleNum", type=int, default=100, help="The number of LF in training set")
parser.add_argument("--patchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--measurementNum", type=int, default=2, help="The number of measurements")
parser.add_argument("--angResolution", type=int, default=7, help="The angular resolution of original LF")
parser.add_argument("--channelNum", type=int, default=1, help="The channel number of input LF")
parser.add_argument("--epochNum", type=int, default=10000, help="The number of epoches")
parser.add_argument("--summaryPath", type=str, default='./', help="Path for saving training log ")
parser.add_argument("--dataName", type=str, default='Lytro', help="The name of dataset ")
parser.add_argument("--dataPath", type=str, default='/public/mantanguo/Dataset/LFCA/train_LFCA_Kalantari_100.mat', help="Path for loading training data ")

opt = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler('{}_m{}_{}_1-5.log'.format(opt.dataName, opt.measurementNum, opt.learningRate))
log.addHandler(fh)
logging.info(opt)

if __name__ == '__main__':

    # SetupSeed(1)
    savePath = './model/{}_m{}_3stage_{}_1-5.pth'.format(opt.dataName, opt.measurementNum, opt.learningRate)
    lfDataset = LFDataset(opt)
    dataloader = DataLoader(lfDataset, batch_size=opt.batchSize, shuffle=True, num_workers = 0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
   

    model=MainNet(opt)
    model._modules['proj_init'].weight.data[model._modules['proj_init'].weight.data<0.0]=0.0
    model._modules['proj_init'].weight.data[model._modules['proj_init'].weight.data>1.0]=1.0

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
            lf=sample['lf']
            lf = to_device(lf,device) # label:[u v c x y] 
            
            estimatedLF=model(lf)
            loss = criterion(estimatedLF,lf)
            lossSum += loss.item()
            
            writer.add_scalar('loss', loss, opt.sampleNum//opt.batchSize*epoch+batch)
            log.info("Time: %s Epoch: %d Batch: %d Loss: %.6f" %(datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S'), epoch,batch,loss.item()))
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model._modules['proj_init'].weight.data[model._modules['proj_init'].weight.data<0.0]=0.0
            model._modules['proj_init'].weight.data[model._modules['proj_init'].weight.data>1.0]=1.0
            
        torch.save(model.state_dict(),savePath)
        log.info("Epoch: %d Loss: %.6f" %(epoch,lossSum/len(dataloader)))
        scheduler.step()

        #Record the training loss
        lossLogger['Epoch'].append(epoch)
        lossLogger['Loss'].append(lossSum/len(dataloader))
        plt.figure()
        plt.title('Loss')
        plt.plot(lossLogger['Epoch'],lossLogger['Loss'])
        plt.savefig('{}_m{}_{}_1-5.jpg'.format(opt.dataName, opt.measurementNum, opt.learningRate))
        plt.close()
    

