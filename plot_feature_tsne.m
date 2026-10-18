function Y = plot_feature_tsne(features, meta, opts)
%PLOT_FEATURE_TSNE Reduce a feature matrix to 2-D with t-SNE and render
%it as an image scatter: each point is drawn as a small thumbnail of its
%own source crop rather than a plain dot.
%
%   With no ground-truth labels available yet, a coloured scatter can't
%   show class separation directly (there's nothing to colour by). An
%   image scatter sidesteps that: if the autoencoder's feature space
%   groups visually similar glyphs near each other, that's visible by
%   eye directly from the thumbnails, no labels required.
%
%   Y = PLOT_FEATURE_TSNE(FEATURES, META) where FEATURES is
%   nFeatures x nSamples (as returned by encode()) and META is the
%   struct array from load_crop_dataset (needs a .Path field).
%
%   Name-value opts (as a struct): Thumb (thumbnail pixel size, default
%   18), Canvas (output canvas size, default 1400), Perplexity (t-SNE
%   perplexity, default 30).

if nargin < 3
    opts = struct();
end
if ~isfield(opts, 'Thumb'); opts.Thumb = 18; end
if ~isfield(opts, 'Canvas'); opts.Canvas = 1400; end
if ~isfield(opts, 'Perplexity'); opts.Perplexity = 30; end

Y = tsne(features', 'Perplexity', opts.Perplexity);

lo = min(Y, [], 1);
hi = max(Y, [], 1);
span = max(hi - lo, eps);

canvasSize = opts.Canvas;
thumb = opts.Thumb;
margin = thumb;
usable = canvasSize - 2 * margin;

canvas = ones(canvasSize, canvasSize);
n = size(Y, 1);

for i = 1:n
    px = round(margin + (Y(i, 1) - lo(1)) / span(1) * usable);
    py = round(margin + (Y(i, 2) - lo(2)) / span(2) * usable);

    img = imread(meta(i).Path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2double(img);
    [h, w] = size(img);
    s = max(h, w);
    sq = ones(s, s);
    r0 = floor((s - h) / 2) + 1;
    c0 = floor((s - w) / 2) + 1;
    sq(r0:r0 + h - 1, c0:c0 + w - 1) = img;
    t = imresize(sq, [thumb, thumb]);

    r1 = max(1, py - floor(thumb / 2));
    c1 = max(1, px - floor(thumb / 2));
    r2 = min(canvasSize, r1 + thumb - 1);
    c2 = min(canvasSize, c1 + thumb - 1);
    if r2 <= r1 || c2 <= c1
        continue;
    end
    patch = canvas(r1:r2, c1:c2);
    tCrop = t(1:(r2 - r1 + 1), 1:(c2 - c1 + 1));
    canvas(r1:r2, c1:c2) = min(patch, tCrop); % darker (ink) wins on overlap
end

figure('Name', 't-SNE feature map');
imshow(canvas);
title(sprintf('%d glyph crops, t-SNE of %d-d autoencoder features', n, size(features, 1)));
end
