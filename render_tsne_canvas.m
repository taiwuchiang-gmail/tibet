function canvas = render_tsne_canvas(Y, meta, opts)
%RENDER_TSNE_CANVAS Draw a precomputed 2-D embedding as an image scatter
%(each point shown as a thumbnail of its own source crop) into a
%specific numbered figure.
%
%   CANVAS = RENDER_TSNE_CANVAS(Y, META, OPTS) where Y is nSamples x 2
%   (e.g. from tsne()), META is the struct array with a .Path field per
%   sample, and OPTS has fields (all optional):
%       Thumb, Canvas - thumbnail and canvas size in pixels
%       FigNum, Title - target figure number and title
%       Images        - sz x sz x 1 x nSamples crops already in memory
%                       (background 1, ink 0); used instead of reading
%                       every META(i).Path from disk
%       Colors        - nSamples x 3 RGB in [0,1]: tint each thumbnail's
%                       ink with its row's color (e.g. by class label)
%                       instead of drawing it gray
%   CANVAS is the full-resolution image (H x W, or H x W x 3 with
%   Colors), for saving with imwrite: figure export downsamples it.
%
%   Split out of plot_feature_tsne.m so a cached/precomputed embedding
%   can be re-rendered (e.g. into its own numbered figure alongside
%   other runs for comparison) without recomputing t-SNE.

if nargin < 3
    opts = struct();
end
if ~isfield(opts, 'Thumb'); opts.Thumb = 16; end
if ~isfield(opts, 'Canvas'); opts.Canvas = 1400; end
if ~isfield(opts, 'FigNum'); opts.FigNum = []; end
if ~isfield(opts, 'Title'); opts.Title = ''; end
if ~isfield(opts, 'Images'); opts.Images = []; end
if ~isfield(opts, 'Colors'); opts.Colors = []; end

Y = single(gather(Y));
lo = min(Y, [], 1);
hi = max(Y, [], 1);
span = max(hi - lo, eps('single'));

canvasSize = opts.Canvas;
thumb = opts.Thumb;
margin = thumb;
usable = canvasSize - 2 * margin;

colored = ~isempty(opts.Colors);
canvas = ones(canvasSize, canvasSize, 1 + 2 * colored, 'single');
n = size(Y, 1);

if ~isempty(opts.Images)
    thumbs = max(0, min(1, imresize(single(gather(opts.Images)), [thumb, thumb])));
end
inkColor = 1 - reshape(single(opts.Colors), [], 1, 1, 3);

for i = 1:n
    px = round(margin + (Y(i, 1) - lo(1)) / span(1) * usable);
    py = round(margin + (Y(i, 2) - lo(2)) / span(2) * usable);

    if isempty(opts.Images)
        t = load_square_thumb(meta(i).Path, thumb);
    else
        t = thumbs(:, :, 1, i);
    end
    if colored
        t = 1 - (1 - t) .* inkColor(i, 1, 1, :);
        t = reshape(t, thumb, thumb, 3);
    end

    r1 = max(1, py - floor(thumb / 2));
    c1 = max(1, px - floor(thumb / 2));
    r2 = min(canvasSize, r1 + thumb - 1);
    c2 = min(canvasSize, c1 + thumb - 1);
    if r2 <= r1 || c2 <= c1
        continue;
    end
    patch = canvas(r1:r2, c1:c2, :);
    tCrop = t(1:(r2 - r1 + 1), 1:(c2 - c1 + 1), :);
    canvas(r1:r2, c1:c2, :) = min(patch, tCrop); % darker (ink) wins on overlap
end

if isempty(opts.FigNum)
    figure('Name', 't-SNE feature map');
else
    figure(opts.FigNum);
end
imshow(canvas);
if isempty(opts.Title)
    title(sprintf('%d glyph crops', n));
else
    title(opts.Title);
end
end

function t = load_square_thumb(path, thumb)
img = imread(path);
if size(img, 3) == 3
    img = rgb2gray(img);
end
img = im2single(img);
[h, w] = size(img);
s = max(h, w);
sq = ones(s, s, 'like', img);
r0 = floor((s - h) / 2) + 1;
c0 = floor((s - w) / 2) + 1;
sq(r0:r0 + h - 1, c0:c0 + w - 1) = img;
t = imresize(sq, [thumb, thumb]);
end
