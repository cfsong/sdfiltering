function [ res, scale] = sdfiltering( img, ss, sr, se, niter, maxWid)
if (~exist('ss','var'))
   ss = 4;
end
if (~exist('sr','var'))
   sr = 0.1;
end
% if (~exist('thrd','var'))
%    thrd = 0.15;
% end
if (~exist('se','var'))
   se = 0.1425;
end
if (~exist('niter','var'))
   niter = 3;
end
if (~exist('maxWid','var'))
   maxWid = 13;
end

[h, w, ~] = size(img);

% [dasm, A, L, D] = com_dasm(img, 7, 5, 3);
% % dasm2 = imadjust(dasm);
% figure
% imshow(dasm, 'Colormap', jet); colorbar; %, 
% title('DASM');

% figure
% imshow(L, 'Colormap', jet); %, 
% title('WIV');

% L0 = gpuArray(im2single(img));
% [~, ~, dRTV] = comp_flatness_rotational(L0, ss, se.^2, 30);
% dRTV2 = gather(dRTV);
% dasm2 = mapminmax(dasm, min(dRTV2(:)), max(dRTV2(:)));
% figure
% imshow(dasm2, 'Colormap', jet); colorbar; %, 
% title('DASM');

% figure
% imshow(dRTV, 'Colormap', jet);  %,  colorbar;
% title('dRTV');

sigma_s = ss; % gaussian kernel for directional RTV
sigma_final = sigma_s*1.5; % gaussian kernel for joint bilateral filtering
sigma_r = sr.*sqrt(size(img, 3)); % color weights kernel for joint bilateral filtering
% rtDeltLmt = 0.35;
% sigma_e = se.^2;
% division = maxWid;
L0 = gpuArray(im2single(img));
L = L0;
% img0 = img;
for ii = 1:niter
    disp(strcat('sdfiltering itertion ', num2str(ii), '...'));

    tic;
    [dasm] = com_dasm(L, 7, 5, 3); %, dr, A, Ap, Lp, Dp
    toc;
    % dasm2 = imadjust(dasm);
    figure %(ii*3+1)
    imshow(dasm, 'Colormap', jet); %, colorbar;
    title('DASM');
%     fname = strcat('DASM_', num2str(ii), 'iter_fish2_new2.png');
%     F = getframe;
%     imDASM = frame2im(F);
%     imwrite(imDASM, fname);

    tic;
     scale = det_scales(dasm, maxWid, se);
%    scale = determ_scale(dasm, maxWid, thrd_e, thrd_p); %, 0.03
    toc;
    color = {'yellow','red','magenta','green','blue', 'cyan', 'white'};
    scales = gather(scale);
%     unq = unique(scales);
    if size(img, 3)>1
        scl = rgb2gray(img);
    else
        scl = img;
    end
    
%     scl0 = scl;
%     wnd = [100 100];
%     rto = 3.0;
%     stPts = [60 330; 195 35; 318 578; 470 175];
%     numAmp = size(stPts, 1);

    interV = 10;
    sp1 = 4:interV:size(scales,1);
    sp2 = 10:interV:size(scales,2);
    [x, y] = meshgrid(sp2, sp1);
    U0 = scales(sp1, sp2);
    U = reshape(U0, size(U0,1)*size(U0,2), 1);
    C = color(mod(U-1,7)+1);
    R = U .* 2;
    Pos = [x(:), y(:), R];
    scl = insertShape(scl, 'Circle', Pos, 'Color',C);

%     mask = zeros(size(scales));
%     mask(sp1, sp2) = 1;
%     for i=1:numAmp
%         stPt = stPts(i,:);
%         edPt = stPt + wnd;
%         Amp = imresize(scl0(stPt(1):edPt(1),stPt(2):edPt(2)), rto);
%         
%         r = [stPt(1),edPt(1),edPt(1),stPt(1); stPt(2), stPt(2),edPt(2),edPt(2)];
%         bw = roipoly(scales,r(2,:),r(1,:));
%         [I J] = find(bw);
%         selc = mask & bw;
%         [xs ys] = find(selc);
%         AA = reshape(scales, [(size(scales,1)*size(scales,2)), 1]);
%         BB = AA(selc, :);
%         Cs = color(mod(BB-1,7)+1);
%         Rs = BB .* (2*rto);
%         Pos2 = [(ys-J(1))*rto+1, (xs-I(1))*rto+1, Rs];
%         Amp = insertShape(Amp, 'Circle', Pos2, 'Color',Cs, 'LineWidth',2);
%         figure, imshow(Amp);
% %         fname11 = strcat('kMap_Amp_', num2str(i), '_fish2.png');
% %         imwrite(Amp, fname11);
% 
%         scl = insertShape(scl, 'Rectangle',[stPt(2),stPt(1),wnd(1),wnd(2)],'LineWidth',4,'Color','r');
%     end

    figure
    imshow(scl);
%     fname1 = strcat('kMap_', num2str(ii), 'iter_fish2_new2.png');
%     imwrite(scl, fname1);

    tic;
    r_L = gaussian_varying_scale(L, scale);
    toc;
%     fname2 = strcat('gMap_', num2str(ii), 'iter_fish2_new2.png');
%     imwrite(gather(r_L), fname2);

    tic;
    L = blf_2d_gpu(L0, r_L, sigma_final, sigma_r);
    toc;
    figure, imshow(L); drawnow;
    
%     img0 = gather(L);

%     fname3 = strcat('result_', num2str(ii), 'iter_fish_new.png');
%     imwrite(img0, fname3);

%     rtDeltLmt = rtDeltLmt * 0.9;
%     se = se * 0.885;
end

% figure(6), close();
res = gather(L);

end