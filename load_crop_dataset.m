function [X, meta] = load_crop_dataset(rootDirs, targetSize, maxSamples)
%LOAD_CROP_DATASET Load glyph crop PNGs (from segment_page.m) into a
%feature matrix suitable for an autoencoder.
%
%   [X, META] = LOAD_CROP_DATASET(ROOTDIRS, TARGETSIZE, MAXSAMPLES)
%   recursively finds *.png files under each folder in the cell array
%   ROOTDIRS, pads each grayscale crop to a square canvas (preserving
%   its aspect ratio -- a plain resize would stretch tall stacked
%   syllables and flat vowel marks into the same box and destroy the
%   shape differences that matter for recognition), resizes to
%   TARGETSIZE x TARGETSIZE, and returns:
%       X    - (TARGETSIZE^2) x N matrix, one normalized [0,1] column
%              per image, N = min(number found, MAXSAMPLES)
%       meta - struct array with fields Path, OrigHeight, OrigWidth
%
%   If more than MAXSAMPLES images are found, a random subset is used
%   (for tractable autoencoder / t-SNE runtime on exploratory runs).
%
%   X is returned on the CPU, in single: each crop is a different size,
%   so the per-image pad/resize can't be batched, and per-image GPU
%   transfers would cost more than the resize itself. Callers move the
%   finished X to the GPU in one transfer.

if nargin < 3
    maxSamples = Inf;
end

paths = {};
for i = 1:numel(rootDirs)
    d = dir(fullfile(rootDirs{i}, '**', '*.png'));
    for k = 1:numel(d)
        paths{end + 1} = fullfile(d(k).folder, d(k).name); %#ok<AGROW>
    end
end

n = numel(paths);
if n == 0
    error('load_crop_dataset:empty', 'No .png files found under the given folders.');
end

if n > maxSamples
    rng(0); % reproducible sampling across runs
    idx = randperm(n, maxSamples);
    paths = paths(idx);
    n = maxSamples;
end

X = zeros(targetSize * targetSize, n, 'single');
meta(n, 1) = struct('Path', '', 'OrigHeight', 0, 'OrigWidth', 0);

for i = 1:n
    img = imread(paths{i});
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2single(img);
    [h, w] = size(img);

    s = max(h, w);
    canvas = ones(s, s, 'like', img); % paper background is light; matches crop background
    r0 = floor((s - h) / 2) + 1;
    c0 = floor((s - w) / 2) + 1;
    canvas(r0:r0 + h - 1, c0:c0 + w - 1) = img;

    resized = imresize(canvas, [targetSize, targetSize]);
    X(:, i) = resized(:);

    meta(i).Path = paths{i};
    meta(i).OrigHeight = h;
    meta(i).OrigWidth = w;
end
end
