function lf4d = eslf2LF4D(eslf,uv_size)

[height,width,channels] = size(eslf);
height = height/uv_size;
width = width/uv_size;
lf4d = reshape(eslf, [uv_size, height, uv_size, width, channels]);
lf4d = permute(lf4d, [1,3,2,4,5]);%u,v,x,y,c