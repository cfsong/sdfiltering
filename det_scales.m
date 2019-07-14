function scale = det_scales(sm, psz, thrd, sigma_e) % , rtDeltLmt, singleVar, totalVar, meanVar, orgVar
if (~exist('psz','var'))
   psz = 15;
end
if (~exist('thrd','var'))
   thrd = 0.385;
end

if (~exist('sigma_e','var'))
   sigma_e = 0.11;
end

[h, w, ~] = size(sm);
se = 2*sigma_e*sigma_e;
fr = ceil((psz-1)*0.5);
% thrsd = thrd;
wd = fr*2 + 1;

nonzeros = fr*(fr*2+1);
entry = 1.0 / nonzeros;
Hup = zeros(wd, wd, 'gpuArray'); 
Hup(1:fr, 1:fr*2+1) = entry;
Hdown = zeros(wd, wd, 'gpuArray');
Hdown(fr+2:fr*2+1, 1:fr*2+1) = entry;
Hleft = zeros(wd, wd, 'gpuArray');
Hleft(1:fr*2+1, 1:fr) = entry;
Hright = zeros(wd, wd, 'gpuArray');
Hright(1:fr*2+1, fr+2:fr*2+1) = entry;

up = imfilter(sm, Hup, 'symmetric');
down = imfilter(sm, Hdown, 'symmetric');
left = imfilter(sm, Hleft, 'symmetric');
right = imfilter(sm, Hright, 'symmetric');
Sgn = cat(3, up, down, left, right);
maxSgn = max(Sgn, [], 3);

% cw = ceil((fr+3)/2.0);
% bg = fr+1+(cw-1)/2;
% ed = fr+1-(cw-1)/2;
% nonzeros = fr*fr;
% nonzeros2 = fr*cw;
% entry = 1.0 / nonzeros;
% entry2 = 1.0 / nonzeros2;
% HupL = zeros(wd, wd, 'gpuArray'); 
% HupL(1:fr, 1:fr) = entry;
% HupC = zeros(wd, wd, 'gpuArray'); 
% HupC(1:fr, bg:ed) = entry2;
% HupR = zeros(wd, wd, 'gpuArray'); 
% HupR(1:fr, fr+2:fr*2+1) = entry;
% HctL = zeros(wd, wd, 'gpuArray'); 
% HctL(bg:ed, 1:fr) = entry2;
% HctR = zeros(wd, wd, 'gpuArray'); 
% HctR(bg:ed, fr+2:fr*2+1) = entry2;
% HdownL = zeros(wd, wd, 'gpuArray');
% HdownL(fr+2:fr*2+1, 1:fr) = entry;
% HdownC = zeros(wd, wd, 'gpuArray');
% HdownC(fr+2:fr*2+1, bg:ed) = entry2;
% HdownR = zeros(wd, wd, 'gpuArray');
% HdownR(fr+2:fr*2+1, fr+2:fr*2+1) = entry;
% 
% upL = imfilter(sm, HupL, 'symmetric');
% upC = imfilter(sm, HupC, 'symmetric');
% upR = imfilter(sm, HupR, 'symmetric');
% ctL = imfilter(sm, HctL, 'symmetric');
% ctR = imfilter(sm, HctR, 'symmetric');
% downL = imfilter(sm, HdownL, 'symmetric');
% downC = imfilter(sm, HdownC, 'symmetric');
% downR = imfilter(sm, HdownR, 'symmetric');
% Sgn = cat(3, upL, upC, upR, ctL, ctR, downL, downC, downR);
% maxSgn = max(Sgn, [], 3);

% tic;
% scale = arrayfun(@sgn2scale, sm, maxSgn, thrd, fr, se);
% toc;
% tic;
sgn2 = bsxfun(@times, maxSgn, maxSgn);
scl = round( exp(-sgn2./se).*fr );
scale = max(1.0, scl);
scale(sm>thrd) = 1.0;
% toc;

% p_sm = padarray(sm, [fr fr], 'symmetric'); % replicate
% scale = ones(h,w); %zeros
% eps = 0.0000001;
% for i = 1:h
%     for j = 1:w
%         center = p_sm(i+fr, j+fr);
%         if (center > thrd)
%             continue;
%         end
%         
%         up = p_sm(i:i+fr-1, j:j+fr*2);
%         down = p_sm(i+fr+1:i+fr*2, j:j+fr*2);
%         left = p_sm(i:i+fr*2, j:j+fr-1);
%         right = p_sm(i:i+fr*2, j+fr+1:j+fr*2);
%         drc = [mean(up(:)) mean(down(:)) mean(left(:)) mean(right(:)) ];
%         maxSgn = max(drc);
%         scl = round( fr * exp(-maxSgn*maxSgn/se) ); % *0.5
%         
%         scale(i,j) = max(1,scl);
% 
%     end
% end

end

function scale = sgn2scale(sm, sgn, thrsd, fr, se) %
if (sm > thrsd)
    scale = single(1.0);
else
    scl = round( fr * exp(-sgn*sgn/se) );
    scale = single(max(1.0, scl));
end
end
