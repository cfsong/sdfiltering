clc;
% clear all;
close all;

img = imread('fish2.jpg'); % GreekMasks.jpg fish.bmp UnicornPhoenix.bmp
figure(16)
imshow(img);

ss = 4;
sr = 0.0985; % 0.05 recommended for other examples 
se = 0.09;   % 0.03 recommended for other examples
niter = 4;
maxWid = 13;
[res, scale] = sdfiltering(img, ss, sr, se, niter, maxWid);
% figure
% imshow(res);
% title('Result');

fname = 'fish2_result.png';
imwrite(res, fname);
