clear; close all;

%% path
dataset_list = {'Stanford','Kalantari'};
folder_list = {...
    '.\Dataset_Stanford_Lytro\',...
    '.\Dataset_kalantari_SIG2016\'};
      
savepath = 'train_all_TIP.h5';
an = 8;

%%initilization
data_HR = zeros(600,600,an,an,1,'uint8');
data_LR_2 = zeros(300,300,an,an,1,'uint8');
data_LR_4 = zeros(150,150,an,an,1,'uint8');
data_size = zeros(2,1,'uint16');
count = 0;

%% read datasets
for i_set = 1:length(dataset_list)
    dataset = dataset_list{i_set};
    folder = folder_list{i_set};
    
    %%% read list
    listname = ['list/train_',dataset,'_TIP.txt'];
    f = fopen(listname);
    C = textscan(f, '%s', 'CommentStyle', '#');
    list = C{1};
    fclose(f);
    
    %%% read lfs
    for i_lf = 1:length(list)
        lfname = list{i_lf};
        
        if strcmp(dataset,'Stanford') || strcmp(dataset,'Kalantari')
            read_path = sprintf('%s/%s.png',folder,lfname);
            
            lf_rgb_hr = read_eslf(read_path,14,an,0);
            lf_rgb_2 = read_eslf(read_path,14,an,2);
            lf_rgb_4 = read_eslf(read_path,14,an,4);
        end
        
        lf_ycbcr_hr = rgb2ycbcr_5d(lf_rgb_hr);
        lf_ycbcr_2 = rgb2ycbcr_5d(lf_rgb_2);
        lf_ycbcr_4 = rgb2ycbcr_5d(lf_rgb_4);
        
        hr = squeeze(lf_ycbcr_hr(:,:,1,:,:)); 
        lr_2 = squeeze(lf_ycbcr_2(:,:,1,:,:));
        lr_4 = squeeze(lf_ycbcr_4(:,:,1,:,:));
        
        H = size(hr,1);
        W = size(hr,2);
        
        count = count +1;
        data_HR(1:H,1:W,:,:,count) = hr;
        data_LR_2(1:H/2,1:W/2,:,:,count) = lr_2;
        data_LR_4(1:H/4,1:W/4,:,:,count) = lr_4;
        data_size(:,count)=[H,W];         
    end
    
end

%% generate data
order = randperm(count);
data_HR = permute(data_HR(:, :, :, :, order),[2,1,4,3,5]); %[h,w,ah,aw,N] -> [w,h,aw,ah,N]  
data_LR_2 = permute(data_LR_2(:, :, :, :, order),[2,1,4,3,5]);
data_LR_4 = permute(data_LR_4(:, :, :, :, order),[2,1,4,3,5]);
data_size = data_size(:,order);  %[2,N]

%% writing to HDF5
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

h5create(savepath, '/img_HR', size(data_HR), 'Datatype', 'uint8'); % width, height, channels, number 
h5create(savepath, '/img_LR_2', size(data_LR_2), 'Datatype', 'uint8'); % width, height, channels, number 
h5create(savepath, '/img_LR_4', size(data_LR_4), 'Datatype', 'uint8');    
h5create(savepath, '/img_size', size(data_size), 'Datatype', 'uint16');   

h5write(savepath, '/img_HR', data_HR);
h5write(savepath, '/img_LR_2', data_LR_2);  
h5write(savepath, '/img_LR_4', data_LR_4);
h5write(savepath, '/img_size', data_size);

h5disp(savepath);
