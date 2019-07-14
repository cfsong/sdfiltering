clc;
clear all;
close all;

img = imread('fish2.jpg'); % GreekMasks.jpg % fish2.jpg fish.bmp UnicornPhoenix.bmp
figure
imshow(img);
title('Input');

ss = 4;
sr = 0.098;
se = 0.098; %0.098
niter = 4;
maxWid = 13;
[res, scale] = sdfiltering(img, ss, sr, se, niter, maxWid);
figure %(25)
imshow(res);
title('Result');

% fname = 'fish2_result.png';
% imwrite(res, fname);
