function out = augment_glyph_old(img)
%AUGMENT_GLYPH Random augmentation of one glyph crop for contrastive
%learning: small rotation, translation, scale jitter, and stroke-width
%(erosion/dilation) jitter, chosen to mimic the kinds of variation real
%scan noise actually produces (slightly bolder/fainter ink, imprecise
%crop boundaries, minor skew) without changing the glyph's identity.
%
%   IMG is HxW, double, in [0,1] with background ~1 (light) and ink ~0
%   (dark), matching load_crop_dataset's convention.

[h, w] = size(img);

% Work in "ink-positive" space (background=0, ink=1) so every geometric
% transform below can fill exposed edges with 0 and get a correct white
% background, instead of the black corners a fill of 0 would produce in
% the original background=1 convention.
ink = 1 - img;

% Small rotation.
angle = (rand() - 0.5) * 16; % +/- 8 deg
ink = imrotate(ink, angle, 'bilinear', 'crop');

% Small translation.
dx = (rand() - 0.5) * 0.15 * w;
dy = (rand() - 0.5) * 0.15 * h;
ink = imtranslate(ink, [dx, dy], 'FillValues', 0);

% Small scale jitter, then re-center back to the original canvas size.
scale = 1 + (rand() - 0.5) * 0.2; % 0.9x - 1.1x
resized = imresize(ink, scale);
ink = center_pad_or_crop(resized, h, w);

% Stroke-width jitter: randomly thicken or thin the ink a little.
r = randi([0, 1]);
if r == 1
    if rand() < 0.5
        ink = imdilate(ink, strel('disk', 1));
    else
        ink = imerode(ink, strel('disk', 1));
    end
end

ink = max(0, min(1, ink));
out = 1 - ink;
end

function out = center_pad_or_crop(img, h, w)
[ih, iw] = size(img);
canvas = zeros(h, w);

if ih <= h
    r0 = floor((h - ih) / 2) + 1;
    rSrc = 1:ih;
    rDst = r0:r0 + ih - 1;
else
    c0 = floor((ih - h) / 2) + 1;
    rSrc = c0:c0 + h - 1;
    rDst = 1:h;
end

if iw <= w
    c0 = floor((w - iw) / 2) + 1;
    cSrc = 1:iw;
    cDst = c0:c0 + iw - 1;
else
    c0 = floor((iw - w) / 2) + 1;
    cSrc = c0:c0 + w - 1;
    cDst = 1:w;
end

canvas(rDst, cDst) = img(rSrc, cSrc);
out = canvas;
end
