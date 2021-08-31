%
% Mattia Rossi (mattia.rossi@epfl.ch)
% Signal Processing Laboratory 4 (LTS4)
% Ecole Polytechnique Federale de Lausanne (Switzerland)
% Modified
%
function LR = high2low(HR, factor, sigmaNoise)
% "high2low" down-samples the light field views in "Z" by "factor" and adds
% Gaussian random noise with standard deviation "sigmaNoise".


% ==== Check the input type ===============================================

%if ~isa(lf2col(HR), 'uint8')
    
%    error('Input light field must have uint8 views !!!\n\n');
    
%end

% ==== Ligth field parameters =============================================

% Angular resolution.
vRes = size(HR, 1);
hRes = size(HR, 2);

% Spatial resolution.
yRes = size(HR, 1);
xRes = size(HR, 2);

% Channels number (gray scale or RGB).
HR = HR(:,:,:);
channels = size(HR, 3);

% ==== Crops "Z" ==========================================================

% Each view of "Z" is cropped at the right and bottom sides, such that
% their dimensions are multiples of "factor".

% Computes the LR dimensions of the views.
yResLR = floor(yRes / factor);
xResLR = floor(xRes / factor);

% New spatial resolution.
yResHR = factor * yResLR;
xResHR = factor * xResLR;

% Performs cropping.     
HR = HR(1:yResHR, 1:xResHR, :);

% ==== Blurs and Decimates "ZHR" ==========================================

% Blur matrix for a single view and channel.
B = blurmat(yResHR, xResHR, factor);

% Decimation matrix for a single view and channel.
D = decimat(yResHR, xResHR, factor);

% Blurring and decimation matrix for a single view and channel.
DB = D * B;

% Blurring and decimation matrix for a single view and ALL its channels.
DBch = kron(speye(channels), DB);

% Number of pixels in each LR view.
n = yResLR * xResLR * channels;

% Blurs, decimates, and adds noise to each view, separately.
       
auxLR = (DBch * double(HR(:)));
auxLR = single(auxLR);

if (mod(channels, 64) == 0)
    LR = reshape(auxLR, [yResLR, xResLR, 1, channels]);
else
    LR = reshape(auxLR, [yResLR, xResLR, channels]);
end

end

