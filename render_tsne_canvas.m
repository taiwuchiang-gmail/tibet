function render_tsne_canvas(Y, meta, opts)
%RENDER_TSNE_CANVAS Draw a precomputed 2-D embedding as an image scatter
%(each point shown as a thumbnail of its own source crop) into a
%specific numbered figure.
%
%   RENDER_TSNE_CANVAS(Y, META, OPTS) where Y is nSamples x 2 (e.g. from
%   tsne()), META is the struct array with a .Path field per sample, and
%   OPTS has fields Thumb, Canvas, FigNum, Title (all optional).
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
    canvas(r1:r2, c1:c2) = min(patch, tCrop);
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
