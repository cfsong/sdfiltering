function [dasm, xi, A, L, D] = com_dasm(img, rds, rdsL, sigmaL)
if (~exist('rds','var'))
   rds = 7;
end
if (~exist('rdsL','var'))
   rdsL = 5;
end
if (~exist('sigmaL','var'))
   sigmaL = 3;
end
eps = 0.0000001;

[h, w, ~] = size(img);
fr = rds;
pl = fr+1;
pr = fr+w;
pu = fr+1;
pb = fr+h;

if size(img, 3)==3
    lab = rgb2lab(gather(img));
    L = lab(:,:,1)./100.0;
elseif size(img, 3)==1
    L = im2double(img);
end

L0 = gpuArray(im2single(L));

% [Ix, Iy] = gradient(L0);
H = [-1 -2 -1;0 0 0 ;1 2 1];%
Ix = filter2(H', L0, 'same');
Iy = filter2(H, L0, 'same');
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy2 = abs(bsxfun(@times, Ix, Iy));

H = fspecial('average', rds*2+1);
Ixx = imfilter(Ix2, H, 'replicate');
Iyy = imfilter(Iy2, H, 'replicate');
Ixy = imfilter(Ixy2, H, 'replicate');

IIxy = bsxfun(@times,Ixy,Ixy);
det = bsxfun(@minus, bsxfun(@times,Ixx,Iyy), IIxy);
tr = bsxfun(@plus, Ixx, Iyy);
sqr = sqrt(bsxfun(@minus, bsxfun(@times,tr, tr), det*4));
eig1 = bsxfun(@plus,tr,sqr)*0.5;
A = bsxfun(@rdivide, sqr, (tr+eps));
% eig2 = bsxfun(@minus,tr,sqr)*0.5;
% egPr = bsxfun(@times, eig1, eig2).^0.5;
% cmb = bsxfun(@times, sqr, egPr);
% tr2 = bsxfun(@times,tr,tr)+eps;
% A = bsxfun(@rdivide, cmb, tr2);
% A_ = (sqr*(eig1*eig2).^0.5)/(tr*tr+eps);
xi_ = bsxfun(@minus, eig1, Ixx);
xi = cat(3, Ixy, xi_);
mg = vecnorm(xi, 2, 3);
xn = bsxfun(@rdivide, xi, mg);

p_A = padarray(A, [fr fr], 'symmetric');
p_xn = padarray(xn, [fr fr], 'symmetric');
D = zeros(h,w, 'gpuArray');
A_sum = zeros(h,w, 'gpuArray');
for y = -fr:fr
    for x = -fr:fr
        xi_tmp = bsxfun(@times, p_xn(pu+y:pb+y, pl+x:pr+x, :), p_A(pu+y:pb+y, pl+x:pr+x));
        D = D + abs( bsxfun(@times, xi_tmp(:,:,1), xn(:,:,1)) + bsxfun(@times, xi_tmp(:,:,2), xn(:,:,2)) );
        A_sum = A_sum + p_A(pu+y:pb+y, pl+x:pr+x);
    end
end
D = bsxfun(@rdivide, D, max(A_sum, eps));

Lx = imgaussfilt(Ix, sigmaL, 'FilterSize',rdsL*2+1, 'Padding', 'symmetric');
Ly = imgaussfilt(Iy, sigmaL, 'FilterSize',rdsL*2+1, 'Padding', 'symmetric');
L = abs(Lx) + abs(Ly);

dasm0 = A .* L .* D;

[mn, mx]=imrange(dasm0);
dasm = (dasm0-mn) ./ (mx-mn);

end