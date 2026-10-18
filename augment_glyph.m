function out = augment_glyph(imgs)
%AUGMENT_GLYPH Random augmentation of glyph crops for contrastive
%learning: small rotation, translation, scale jitter, and stroke-width
%(erosion/dilation) jitter, chosen to mimic the kinds of variation real
%scan noise actually produces (slightly bolder/fainter ink, imprecise
%crop boundaries, minor skew) without changing the glyph's identity.
%
%   IMGS is HxW or HxWx1xN, single, in [0,1] with background ~1 (light)
%   and ink ~0 (dark), matching load_crop_dataset's convention. Each of
%   the N samples gets its own independent random parameters. The whole
%   batch is processed with array ops only (one batched bilinear warp
%   for rotation+translation+scale), so it runs on the GPU when IMGS is a
%   gpuArray and stays in IMGS's precision.

[h, w, c, n] = size(imgs);
assert(c == 1, 'augment_glyph expects single-channel crops.');
assert((h + 2) * (w + 2) * n < flintmax('single'), ...
    'Batch too large for exact single-precision gather indices.');

% Work in "ink-positive" space (background=0, ink=1) so out-of-bounds
% samples fill with 0 and come out as white background, instead of the
% black corners a fill of 0 would produce in the background=1 convention.
ink = 1 - imgs;

angle = (rand(1, 1, 1, n, 'like', imgs) - 0.5) * (16 * pi / 180); % +/- 8 deg
dx = (rand(1, 1, 1, n, 'like', imgs) - 0.5) * 0.15 * w;
dy = (rand(1, 1, 1, n, 'like', imgs) - 0.5) * 0.15 * h;
scale = 1 + (rand(1, 1, 1, n, 'like', imgs) - 0.5) * 0.2; % 0.9x - 1.1x

% Forward map is rotate about the center, then translate, then scale
% about the center. Sample by inverting it for every output pixel.
cx = (w + 1) / 2;
cy = (h + 1) / 2;
[X, Y] = meshgrid(cast(1:w, 'like', imgs), cast(1:h, 'like', imgs));
u = (X - cx) ./ scale - dx;
v = (Y - cy) ./ scale - dy;
ca = cos(angle);
sa = sin(angle);
xs = cx + ca .* u + sa .* v;
ys = cy - sa .* u + ca .* v;

ink = bilinear_sample(ink, xs, ys);

% Stroke-width jitter: half the samples get a 1px thicken or thin.
jitter = rand(1, 1, 1, n, 'like', imgs) < 0.5;
thicken = rand(1, 1, 1, n, 'like', imgs) < 0.5;
isDil = jitter & thicken;
isEro = jitter & ~thicken;
if any(isDil)
    ink(:, :, :, isDil) = cross_filter(ink(:, :, :, isDil), @max, 0);
end
if any(isEro)
    ink(:, :, :, isEro) = cross_filter(ink(:, :, :, isEro), @min, 1);
end

ink = max(0, min(1, ink));
out = 1 - ink;
end

function out = bilinear_sample(img, xs, ys)
% Batched bilinear interpolation with zero fill outside the image.
[h, w, ~, n] = size(img);
padded = zeros(h + 2, w + 2, 1, n, 'like', img);
padded(2:end-1, 2:end-1, :, :) = img;

inside = xs > 0 & xs < w + 1 & ys > 0 & ys < h + 1;
x0 = min(max(floor(xs), 0), w);
y0 = min(max(floor(ys), 0), h);
fx = xs - x0;
fy = ys - y0;

hp = h + 2;
page = reshape(cast(0:n-1, 'like', img), 1, 1, 1, n) * (hp * (w + 2));
idx = (y0 + 1) + x0 * hp + page;

out = (1 - fx) .* (1 - fy) .* padded(idx) ...
    + (1 - fx) .* fy .* padded(idx + 1) ...
    + fx .* (1 - fy) .* padded(idx + hp) ...
    + fx .* fy .* padded(idx + hp + 1);
out = out .* inside;
end

function out = cross_filter(img, op, padValue)
% Grayscale dilation (op=@max) or erosion (op=@min) with strel('disk', 1),
% i.e. the 3x3 cross. padValue makes the outside neutral for op, matching
% imdilate/imerode border handling.
[h, w, c, n] = size(img);
p = padValue + zeros(h + 2, w + 2, c, n, 'like', img);
p(2:end-1, 2:end-1, :, :) = img;
out = op(op(op(img, p(1:end-2, 2:end-1, :, :)), op(p(3:end, 2:end-1, :, :), ...
    p(2:end-1, 1:end-2, :, :))), p(2:end-1, 3:end, :, :));
end
