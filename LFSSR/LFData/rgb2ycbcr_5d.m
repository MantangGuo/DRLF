function lf_ycbcr = rgb2ycbcr_5d(lf_rgb)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lf_rgb [h,w,3,ah,aw] --> lf_ycbcr [h,w,3,ah,aw]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if length(size(lf_rgb))<5
    error('input must have 5 dimensions');
else
    lf_ycbcr = zeros(size(lf_rgb),'like',lf_rgb);
    for v = 1:size(lf_rgb,4)
        for u = 1:size(lf_rgb,5)
            lf_ycbcr(:,:,:,v,u) = rgb2ycbcr(lf_rgb(:,:,:,v,u));
        end
    end

end
