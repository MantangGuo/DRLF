clear all;
%% Parameter setting
gtPath='.\Dataset_Stanford_Lytro\';
savePath='.\train_noiseLevel_10-20-50_4-11.mat';
sceneClass={'bikes','buildings','cars','flowers_plants','fruits_vegetables','people','people','general'};
chosenSceneEachClass=[[10,19];[10,19];[10,19];[10,19];[10,19];[1,2];[10,17];[10,19]];
% h5Structure.label='/lf';
% h5Structure.input=["/noilf_10","/noilf_20","/noilf_50"];
% h5Structure.savePath=savePath;              
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
        gt=eslf2LF4D(eslf,eslfAngSize); %[u,v,x,y] [0-255]
        curLF=gt(startAngCordi:startAngCordi+angSize-1,...
              startAngCordi:startAngCordi+angSize-1,...
              :,:); %u,v,x,y
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
%% shuffle and save the label and input
order=randperm(count);
lf=lf(:,:,:,:,order);
noilf_10=noilf_10(:,:,:,:,order);
noilf_20=noilf_20(:,:,:,:,order);
noilf_50=noilf_50(:,:,:,:,order);

save(savePath,'lf','noilf_10','noilf_20','noilf_50','LF_name','-v6');

% Data2H5(count, h5Structure,lf,noilf_10,noilf_20,noilf_50);


    


