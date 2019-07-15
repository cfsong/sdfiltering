function [ res, scale] = sdfiltering( img, ss, sr, se, niter, maxWid)
if (~exist('ss','var'))
   ss = 4;
end
if (~exist('sr','var'))
   sr = 0.05;
end
if (~exist('se','var'))
   se = 0.05;
end
if (~exist('niter','var'))
   niter = 3;
end
if (~exist('maxWid','var'))
   maxWid = 13;
end

sigma_s = ss;
sigma_final = sigma_s*1.5;
sigma_r = sr.*sqrt(size(img, 3));

L0 = gpuArray(im2single(img));
L = L0;

for ii = 1:niter
    disp(strcat('sdfiltering itertion', num2str(ii), '...'));
    
    dasm = com_dasm(L, 7, 5, 3);
    
    scale = det_scales(dasm, maxWid, se);
    
    r_L = gaussian_varying_scale(L, scale);
    
    L = blf_2d_gpu(L0, r_L, sigma_final, sigma_r);
    
    tl = strcat(num2str(ii), ' itertion(s)');
    figure(16), imshow(L);
    title(tl);
    
    se = se * 0.865;
end

res = gather(L);

end