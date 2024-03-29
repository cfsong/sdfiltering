% This file is from the paper:
% [Scale-aware Structure-Preserving Texture Filtering]
% Computer Graphics Forum (special issue on Pacific Graphics 2016), Vol. 35, No. 7, 2016.
% Junho Jeon, Hyunjoon Lee, Henry Kang, Seungyong Lee

function R = gaussian_varying_scale(L, ss_map)

ss_max = gather(max(ss_map(:)));

fr = cast(ceil(3*ss_max), 'double');

iss = max(eps, 1 ./ ss_map.^2 ); % ./ ss_map

% iss = 1./SS./SS;

[h, w, ~] = size(L);
p_L = padarray(L, [fr fr], 'symmetric');
pl = fr+1;
pr = fr+w;
pu = fr+1;
pb = fr+h;

R = zeros(size(L), 'like', L);
w_sum = zeros(h, w, 'like', L);

for y = -fr:fr
    for x = -fr:fr
        %     d2 = y.^2 + x.^2;
        if x == 0 && y == 0
            w_s = 1;
        else
            w_s = max(0, exp((-0.5*(x*x+y*y))*iss));
        end
        R = R + bsxfun(@times, p_L(pu+y:pb+y, pl+x:pr+x, :), w_s);
        w_sum = w_sum + w_s;
    end
end

R = bsxfun(@rdivide, R, max(w_sum, eps));

end