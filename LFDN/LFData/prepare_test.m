clear all;
%% Parameter setting
gtPath='.\Dataset_Stanford_Lytro\';
savePath='.\test_noiseLeve_10-20-50_4-11.mat';
sceneClass={'bikes','buildings','cars','flowers_plants','fruits_vegetables','people'};
chosenSceneEachClass=[[2,6];[21,25];[21,25];[21,25];[2,6];[5,9]];             
eslfAngSize=14;
startAngCordi=4;
angSize=8;
noiseLevel=[10,20,50];
viewSize=[375,540];

%% Creat Data 
count=0;
LF_name = {};
for i=1:size(sceneClass,2)
    for j=chosenSceneEachClass(i,1):chosenSceneEachClass(i,2)
        count=count+1;

%eslf-----LF4D
        eslf=rgb2gray(im2uint8(imread([gtPath,sceneClass{i},'_',num2str(j),'_eslf.png'])));        
        gt=eslf2LF4D(eslf,eslfAngSize); %[u,v,x,y,c] [0-255]
        curLF=gt(startAngCordi:startAngCordi+angSize-1,...
              startAngCordi:startAngCordi+angSize-1,...
              :,:); %u,v,x,y,c
        curLF=curLF(:,:,1:viewSize(1),1:viewSize(2));
        
%add AWGN with different standard variant
        lf(:,:,:,:,count)=curLF;
        noilf_10(:,:,:,:,count)=uint8(double(curLF)+noiseLevel(1)*randn(size(curLF)));
        noilf_20(:,:,:,:,count)=uint8(double(curLF)+noiseLevel(2)*randn(size(curLF)));
        noilf_50(:,:,:,:,count)=uint8(double(curLF)+noiseLevel(3)*randn(size(curLF)));
        LF_name= cat(1,LF_name,abs([sceneClass{i},'_',num2str(j),'_eslf']));
        
        disp(size(lf));
        disp(size(noilf_10));
        disp(size(noilf_20));
        disp(size(noilf_50));
        disp(size(LF_name));
    end
end
save(savePath,'lf','noilf_10','noilf_20','noilf_50','LF_name','-v6');


    


