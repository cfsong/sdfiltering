function [dasm, xi, A, L, D] = com_dasm(img, rds, rdsL, sigmaL, sigma_e)
if (~exist('rds','var'))
   rds = 7;
end
if (~exist('rdsL','var'))
   rdsL = 5;
end
if (~exist('sigmaL','var'))
   sigmaL = 3;
end
if (~exist('sigma_e','var'))
   sigma_e = 0.008 .^2;
end

[h, w, ~] = size(img);
fr = rds;
pl = fr+1;
pr = fr+w;
pu = fr+1;
pb = fr+h;

if size(img, 3)==3
    lab = rgb2lab(gather(img));
    L = lab(:,:,1)./100.0;
%     hsv = rgb2hsv(img);
%     L= hsv(:,:,3);
elseif size(img, 3)==1
    L = im2double(img);
end
% figure
% imshow(L, []); % 0 100
% title('Lab Channel');
% bw = edge(L, 'canny');
% figure, imshow(bw);

L0 = gpuArray(im2single(L));

% H = [-1 -2 -1;0 0 0 ;1 2 1];%
% Ix = filter2(H', L, 'same');
% Iy = filter2(H, L, 'same');
[Ix, Iy] = gradient(L0);
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy2 = abs(bsxfun(@times, Ix, Iy)); % 

% Ixx = zeros(h,w); % size(L), 'like', L
% Iyy = zeros(h,w); % size(L), 'like', L
% Ixy = zeros(h,w); % size(L), 'like', L

% tic
% p_Ix2 = padarray(Ix2, [fr fr], 'symmetric');
% p_Ixy2 = padarray(Ixy2, [fr fr], 'symmetric');
% p_Iy2 = padarray(Iy2, [fr fr], 'symmetric');
% for y = -fr:fr
%     for x = -fr:fr
%         Ixx = Ixx + p_Ix2(pu+y:pb+y, pl+x:pr+x);
%         Iyy = Iyy + p_Iy2(pu+y:pb+y, pl+x:pr+x);
%         Ixy = Ixy + p_Ixy2(pu+y:pb+y, pl+x:pr+x);
%     end
% end
% toc

% nlfilter is very slow!
% tic
% fun = @(x) sum(x(:)); 
% Ixx = nlfilter(Ix2, [rds*2+1 rds*2+1], fun);
% Iyy = nlfilter(Iy2, [rds*2+1 rds*2+1], fun);
% Ixy = nlfilter(Ixy2, [rds*2+1 rds*2+1], fun);
% toc

% H = fspecial('average', rds*2+1);
% % step = (rds*2+1)*(rds*2+1);
% Ixx = imfilter(Ix2, H, 'replicate'); % *step
% Iyy = imfilter(Iy2, H, 'replicate'); % *step
% Ixy = imfilter(Ixy2, H, 'replicate'); % *step
Ixx = imgaussfilt(Ix2, 5, 'FilterSize',rdsL*2+1, 'Padding', 'symmetric');
Iyy = imgaussfilt(Iy2, 5, 'FilterSize',rdsL*2+1, 'Padding', 'symmetric');
Ixy = imgaussfilt(Ixy2, 5, 'FilterSize',rdsL*2+1, 'Padding', 'symmetric');

eps = 0.0000001;

% A0 = zeros(h,w); % size(L), 'like', L
% xi0 = zeros(h,w, 2); % size(L), 'like', L
% A = zeros(h,w, 'gpuArray'); % size(L), 'like', L
% xi = zeros(h,w, 2, 'gpuArray');
% for i=1:h
%     for j=1:w
%         ST = gather([Ixx(i,j), Ixy(i,j); Ixy(i,j), Iyy(i,j)]); % 
%         [V, D] = eig(ST , 'vector'); %
%         A(i,j) = abs((D(2)-D(1))/(D(2)+D(1)+eps));
%         xi(i,j,:) = V(:,1);
%     end
% end
% tic;
% [A, xn1, xn2] = arrayfun(@strucTensor, Ixx, Ixy, Iyy); % xi1, xi2, 
% % xi = cat(3,xi1, xi2);
% xn = cat(3,xn1, xn2);
% toc;
% tic;
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
% mod = sqrt(bsxfun(@plus, IIxy, bsxfun(@times,xi_,xi_)))+eps;
% xn1 = bsxfun(@rdivide,Ixy, mod);
% xn2 = bsxfun(@rdivide, xi_, mod);
% xn = cat(3,xn1, xn2);
% toc;

% tic;
p_A = padarray(A, [fr fr], 'symmetric');
p_xn = padarray(xn, [fr fr], 'symmetric');
D = zeros(h,w, 'gpuArray'); % size(L), 'like', L
% xi_tmp = zeros(h,w, 2); % size(L), 'like', L
A_sum = zeros(h,w, 'gpuArray'); % size(L), 'like', L
for y = -fr:fr
    for x = -fr:fr
        xi_tmp = bsxfun(@times, p_xn(pu+y:pb+y, pl+x:pr+x, :), p_A(pu+y:pb+y, pl+x:pr+x));
        D = D + abs( bsxfun(@times, xi_tmp(:,:,1), xn(:,:,1)) + bsxfun(@times, xi_tmp(:,:,2), xn(:,:,2)) );
        A_sum = A_sum + p_A(pu+y:pb+y, pl+x:pr+x);
%         inPrd_tmp = bsxfun(@plus, bsxfun(@times,p_xn(pu+y:pb+y, pl+x:pr+x,1), xn(:,:,1)), bsxfun(@times, p_xn(pu+y:pb+y, pl+x:pr+x,2), xn(:,:,2)));
%         D = bsxfun(@plus, D, abs(bsxfun(@times, inPrd_tmp, p_A(pu+y:pb+y, pl+x:pr+x))));
%         A_sum = bsxfun(@plus, A_sum, p_A(pu+y:pb+y, pl+x:pr+x));
    end
end
D = bsxfun(@rdivide, D, max(A_sum, eps));
% toc;

% tic
% Lx = zeros(h,w); % size(L), 'like', L
% Ly = zeros(h,w); % size(L), 'like', L
% p_Ix = padarray(Ix, [fr fr], 'symmetric');
% p_Iy = padarray(Iy, [fr fr], 'symmetric');
% iss = max(eps,1.0/(sigmaL*sigmaL));
% fr = rdsL;
% pl = fr+1;
% pr = fr+w;
% pu = fr+1;
% pb = fr+h;
% for y = -fr:fr
%     for x = -fr:fr
%         if x==0 && y==0
%             w_s = 1;
%         else
%             w_s = max(0, exp(-0.5 * ((x*x+y*y)*iss)));
%         end
%         Lx = Lx + bsxfun(@times, p_Ix(pu+y:pb+y, pl+x:pr+x), w_s);
%         Ly = Ly + bsxfun(@times, p_Iy(pu+y:pb+y, pl+x:pr+x), w_s);
%     end
% end
% toc

Lx = imgaussfilt(Ix, sigmaL, 'FilterSize',rdsL*2+1, 'Padding', 'symmetric');
Ly = imgaussfilt(Iy, sigmaL, 'FilterSize',rdsL*2+1, 'Padding', 'symmetric');
L = abs(Lx) + abs(Ly);
dasm0 = A .* L .* D;
% dasm0 = bsxfun(@times, A, bsxfun(@times,L,D));

% dasm = mapminmax(dasm0, 0, 1);
[mn mx]=imrange(dasm0);
dasm = (dasm0-mn) ./ (mx-mn);

% dasm = dasm0 ./ (max(dasm0(:)));
% dasm = imadjust(dasm);

% dasm = dasm0 .^2;
% dasm = exp(-0.5 * sigma_e ./ dasm);

end

function [A_, xn1, xn2] = strucTensor(gxx, gxy, gyy) % xi1, xi2, 
% ST = ([gxx, gxy; gxy, gyy]); %
% [V0, D0] = eig(ST , 'vector'); %
% A_ = abs((D0(2)-D0(1))/(D0(2)+D0(1)+eps));
% xi1 = V0(1,1);
% xi2 = V0(2,1);
det = gxx*gyy-gxy*gxy;
tr = gxx+gyy;
sqr = sqrt(tr*tr-det*4.0);
eig1 = (tr+sqr)*0.5;
% eig2 = (tr-sqr)*0.5;
% lumda = eig1;
A_ = sqr/(tr+eps);
% A_ = (sqr*(eig1*eig2).^0.5)/(tr*tr+eps);
% A_ = det - 0.0215*tr*tr;
% A_ = ((eig1)/0.45).^0.33;
% if eig1 > 0.3
% A_ = 2.0;
% else
%     A_ = 0.75;
% end
 
xi_ = eig1-gxx;
% xi_2 = eig2-gxx;
% xi1 = gxy;
% xi2 = xi_;
mod = sqrt(gxy*gxy+xi_*xi_)+eps;
% mod2 = sqrt(gxy*gxy+xi_2*xi_2)+eps;
xn1 = gxy / mod;
% xn1_2 = gxy / mod2;
xn2 = xi_ / mod;
% xn2_2 = xi_2 / mod2;
% xx = [xn1 xn2]'*eig1 + [xn1_2 xn2_2]'*eig2;
% xi1 = xx(1);
% xi2 = xx(2);

% A_ = abs((D0(2)-D0(1))/(D0(2)+D0(1)+eps));
% xi_ = V0(:,1);

end