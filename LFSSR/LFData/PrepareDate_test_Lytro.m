
clear;close all;

%% params
%%% Stanford
data_folder = '.\Dataset_Stanford_Lytro\';
savepath = 'test_Stanford_General_x4_TIP.mat';
% data_folder = '.\Dataset_Stanford_Lytro\';
% savepath = 'test_Stanford_General_x2_TIP.mat';


scale = 4;
an = 8;

h = 372;
w = 540;

%% initilization
GT_y   = [];
LR_ycbcr = [];
LF_name = {};
count = 0;

data_list = dir(data_folder);
data_list = data_list(3:end);
%% generate data
for k = 1:length(data_list)
    lfname = data_list(k).name;
    read_path = fullfile(data_folder,lfname);
    lf_gt_rgb = read_eslf(read_path, 14, an, 0); %[h,w,3,ah,aw]
    lf_lr_rgb = read_eslf(read_path, 14, an, scale); %[h,w,3,ah,aw]
    
    lf_gt_rgb = lf_gt_rgb(1:h,1:w,:,:,:);
    lf_lr_rgb = lf_lr_rgb(1:h/scale,1:w/scale,:,:,:);
    
    lf_gt_ycbcr = rgb2ycbcr_5d(lf_gt_rgb);
    lf_lr_ycbcr = rgb2ycbcr_5d(lf_lr_rgb);
    
    lf_gt_y = squeeze(lf_gt_ycbcr(:,:,1,:,:)); %[h,w,ah,aw]
%     lf_lr_ycbcr = imresize(lf_gt_ycbcr,1/scale,'bicubic');  %[h/s,w/s,3,ah,aw]
    
    
    GT_y = cat(5,GT_y,lf_gt_y);
    LR_ycbcr = cat(6,LR_ycbcr,lf_lr_ycbcr);
    LF_name = cat(1,LF_name,abs(lfname(1:end-4)));
end

GT_y = permute(GT_y,[5,3,4,1,2]); 
LR_ycbcr = permute(LR_ycbcr,[6,4,5,3,1,2]); 

%% save data
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 
save(savepath, 'GT_y', 'LR_ycbcr', 'LF_name','-v6');

% h5create(savepath,'/GT_y',size(GT_y),'Datatype','uint8');
% h5create(savepath,'/LR_ycbcr',size(LR_ycbcr),'Datatype','uint8');
% h5create(savepath,'/LF_name',size(LF_name),'Datatype','string');
% 
% h5write(savepath, '/GT_y', GT_y);
% h5write(savepath, '/LR_ycbcr', LR_ycbcr);
% h5write(savepath, '/LF_name', LF_name);
% 
% h5disp(savepath);