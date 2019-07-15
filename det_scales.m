function scale = det_scales(sm, psz, sigma_e, thrd)
if (~exist('psz','var'))
   psz = 15;
end
if (~exist('thrd','var'))
   thrd = 0.3;
end
if (~exist('sigma_e','var'))
   sigma_e = 0.11;
end

se = 2*sigma_e*sigma_e;
fr = ceil((psz-1)*0.5);
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

sgn2 = bsxfun(@times, maxSgn, maxSgn);
scl = round( exp(-sgn2./se).*fr );
scale = max(1.0, scl);
scale(sm>thrd) = 1.0;

end
