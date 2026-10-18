function results = segment_page(imagePath, varargin)
%SEGMENT_PAGE Detect a printed page frame, crop its interior, and extract
%glyph-cluster candidates (connected ink blobs) as individual crops.
%
%   RESULTS = SEGMENT_PAGE(IMAGEPATH) segments the page image at
%   IMAGEPATH and returns a struct array, one entry per accepted glyph
%   blob, with fields:
%       BoundingBox    - [x y w h] in the cropped page's coordinates
%       Line           - text line index (1 = topmost line)
%       PositionInLine - left-to-right order within its line
%       FilePath       - path to the saved crop (if SaveCrops is true)
%
%   Unlike the original cc.m prototype, nothing here is tied to one
%   image's pixel dimensions: the page frame is located from the ink
%   itself (largest thin rectangular connected component), and glyph
%   filtering thresholds are derived from the page's own component-size
%   statistics rather than fixed absolute pixel counts. Pages with no
%   printed frame (e.g. photographed manuscripts) fall back to using the
%   whole image.
%
%   Name-value options:
%       'OutputDir'  - folder to save crops into (default: "<image
%                      basename>_crops" next to the source image)
%       'SaveCrops'  - true/false, write crop PNGs (default true)
%       'Visualize'  - true/false, show a figure with detected boxes (default false)
%       'Deskew'     - true/false, correct small page rotation first (default true)

p = inputParser;
addParameter(p, 'OutputDir', '', @(s) ischar(s) || isstring(s));
addParameter(p, 'SaveCrops', true, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Visualize', false, @(x) islogical(x) || isnumeric(x));
addParameter(p, 'Deskew', true, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});
opts = p.Results;

[srcDir, baseName] = fileparts(imagePath);
if strlength(string(opts.OutputDir)) == 0
    outputDir = fullfile(srcDir, baseName + "_crops");
else
    outputDir = string(opts.OutputDir);
end

I = imread(imagePath);
if size(I, 3) == 3
    gray = rgb2gray(I);
else
    gray = I;
end

BW = imbinarize(gray, "global");
ink = pick_ink_mask(BW);

angle = 0;
if opts.Deskew
    angle = estimate_skew(ink);
    if abs(angle) > 0.05
        % Rotate the already-binarized mask directly (nearest-neighbor,
        % so it stays crisp 0/1) rather than rotating the grayscale and
        % re-binarizing afterwards -- bilinear interpolation blurs pixel
        % values across the rotation, and re-running Otsu on that blur
        % can wipe out most of the ink on already low-contrast scans.
        % The grayscale is still rotated (bilinear) separately, purely
        % for saved crop image quality.
        gray = imrotate(gray, angle, "bilinear", "loose");
        ink = imrotate(ink, angle, "nearest", "loose");
    end
end

[frameBox, hasFrame] = find_page_frame(ink);

if hasFrame
    x = round(frameBox(1)); y = round(frameBox(2));
    w = round(frameBox(3)); h = round(frameBox(4));
    cropInk = ink(y:y+h-1, x:x+w-1);
    cropGray = gray(y:y+h-1, x:x+w-1);
else
    cropInk = ink;
    cropGray = gray;
end

[compTable, numLines] = extract_glyph_components(cropInk);

if isempty(compTable)
    warning('segment_page:noComponents', ...
        'No glyph-like components survived filtering for %s.', imagePath);
    results = struct('BoundingBox', {}, 'Line', {}, 'PositionInLine', {}, 'FilePath', {});
    return;
end

if opts.SaveCrops && ~isfolder(outputDir)
    mkdir(outputDir);
end

n = height(compTable);
results(n, 1) = struct('BoundingBox', [], 'Line', [], 'PositionInLine', [], 'FilePath', "");

for k = 1:n
    bb = compTable.BoundingBox(k, :);
    x1 = max(1, floor(bb(1)) - 2);
    y1 = max(1, floor(bb(2)) - 2);
    x2 = min(size(cropGray, 2), ceil(bb(1) + bb(3)) + 2);
    y2 = min(size(cropGray, 1), ceil(bb(2) + bb(4)) + 2);
    glyphImg = cropGray(y1:y2, x1:x2);

    results(k).BoundingBox = bb;
    results(k).Line = compTable.Line(k);
    results(k).PositionInLine = compTable.PositionInLine(k);

    if opts.SaveCrops
        fname = sprintf('%s_line%02d_pos%03d.png', baseName, compTable.Line(k), compTable.PositionInLine(k));
        fpath = fullfile(outputDir, fname);
        imwrite(glyphImg, fpath);
        results(k).FilePath = string(fpath);
    else
        results(k).FilePath = "";
    end
end

if opts.Visualize
    figure('Name', sprintf('segment_page: %s', baseName));
    imshow(cropGray);
    hold on;
    for k = 1:n
        bb = compTable.BoundingBox(k, :);
        rectangle('Position', bb, 'EdgeColor', 'g', 'LineWidth', 1);
    end
    hold off;
    title(sprintf('%d glyph candidates, %d lines, skew %.2f deg', n, numLines, angle));
end

end

%% ---------------------------------------------------------------------
%  Local helpers
%  -----------------------------------------------------------------

function ink = pick_ink_mask(BW)
% Ink is assumed to be the minority class by pixel count (dark strokes
% on a lighter page background). Falls back gracefully if the page scan
% happens to be inverted.
if nnz(BW) <= numel(BW) / 2
    ink = BW;
else
    ink = ~BW;
end
end

function angle = estimate_skew(ink)
% Estimate small page rotation from the dominant near-horizontal /
% near-vertical edge directions.
edges = edge(ink, 'canny');
[H, theta, ~] = hough(edges, 'RhoResolution', 0.5, 'Theta', -45:0.2:44.8);
P = houghpeaks(H, 40, 'threshold', ceil(0.2 * max(H(:))));
if isempty(P)
    angle = 0;
    return;
end
angle = median(theta(P(:, 2)));
end

function cleaned = remove_rule_lines(ink)
% Strip long straight printed rule/divider lines from the ink mask.
% A line-shaped structuring element as long as a healthy fraction of the
% page's own height/width only survives morphological opening if the ink
% under it is itself a long straight run -- true rule lines qualify,
% glyph strokes (short and curved) do not.
[h, w] = size(ink);
vertLen = round(h * 0.5);
horizLen = round(w * 0.3);

rules = false(h, w);
if vertLen >= 3
    rules = rules | imopen(ink, strel('line', vertLen, 90));
end
if horizLen >= 3
    rules = rules | imopen(ink, strel('line', horizLen, 0));
end

cleaned = ink & ~rules;
end

function [frameBox, hasFrame] = find_page_frame(ink)
% Identify the printed border, if any, as the connected component whose
% bounding box spans most of the page but whose ink fills only a thin
% fraction of that bounding box (i.e. an outline, not a filled block).
CC = bwconncomp(ink, 8);
if CC.NumObjects == 0
    frameBox = [];
    hasFrame = false;
    return;
end

s = regionprops(CC, 'BoundingBox', 'Area', 'Perimeter');
bbx = cat(1, s.BoundingBox);
areas = cat(1, s.Area);
perim = cat(1, s.Perimeter);

[imH, imW] = size(ink);
coverage = (bbx(:, 3) ./ imW) .* (bbx(:, 4) ./ imH);
fillRatio = areas ./ (bbx(:, 3) .* bbx(:, 4));

isFrameLike = coverage > 0.5 & fillRatio < 0.35 & perim > 0;

if ~any(isFrameLike)
    frameBox = [];
    hasFrame = false;
    return;
end

candIdx = find(isFrameLike);
[~, best] = max(coverage(candIdx));
idx = candIdx(best);

% Thin rectangular outline: area ~= perimeter * stroke thickness.
thickness = areas(idx) / perim(idx);
inset = max(2, round(thickness * 1.5));

bb = bbx(idx, :);
x = min(max(bb(1) + inset, 1), imW);
y = min(max(bb(2) + inset, 1), imH);
w = max(bb(3) - 2 * inset, 1);
h = max(bb(4) - 2 * inset, 1);
w = min(w, imW - x + 1);
h = min(h, imH - y + 1);

frameBox = [x y w h];
hasFrame = true;
end

function [compTable, numLines] = extract_glyph_components(ink)
% Keep connected components whose size/shape is consistent with the
% page's own glyph-size statistics (not fixed absolute pixel counts),
% then segment text lines from the row-wise projection profile of just
% those glyph pixels.
%
% Printed rule/divider lines (long straight column dividers) are removed
% up front, directionally: a long vertical or horizontal rule line often
% touches glyph ink from several unrelated lines/columns as it crosses
% them, which silently fuses all of that into one giant, page-spanning
% connected component -- one that gets (correctly) rejected by the size
% filter below, taking every real glyph it touched down with it. Rule
% lines are far longer and straighter than any glyph stroke, so opening
% with a long line-shaped structuring element isolates them cleanly
% without disturbing glyph shapes (verified: on the reference page this
% recovered glyphs that were previously vanishing into 900+ px "blobs"
% that were mostly a rule line, moving the kept-component count from 932
% to 947 -- real recovery, not a wild swing in either direction).
%
% (A morphological closing pass was also tried, to rejoin strokes that
% binarization severs mid-glyph, and reverted -- at this scan's
% resolution the gap from a severed stroke and the gap between two
% genuinely different adjacent glyphs are both ~1-2px, so any closing
% radius that bridges one also fuses the other. That needs something
% that looks at more than pixel proximity, e.g. stroke continuity, which
% is a bigger change than a blanket closing operation.)
ink = remove_rule_lines(ink);

% On lower-contrast/faded scans, binarization leaves thousands of
% few-pixel salt-noise specks. Those specks outnumber real glyph
% fragments, which drags the median component area down to almost
% nothing and makes the "relative to median" filter below reject nearly
% everything (including real glyphs). Clearing that noise first, using a
% stroke-width estimate from the page's own ink, keeps the median
% meaningful regardless of scan quality.
ink = denoise_by_stroke_width(ink);

CC = bwconncomp(ink, 4);
if CC.NumObjects == 0
    compTable = table();
    numLines = 0;
    return;
end

s = regionprops(CC, 'Area', 'Extent', 'BoundingBox');
areas = cat(1, s.Area);
extent = cat(1, s.Extent);
bbx = cat(1, s.BoundingBox);
ratio = bbx(:, 3) ./ bbx(:, 4);

medArea = median(areas);

keep = areas > max(6, medArea * 0.08) & areas < medArea * 15 ...
     & extent > 0.08 & ratio > 0.08 & ratio < 12 ...
     & bbx(:, 3) < size(ink, 2) * 0.4 & bbx(:, 4) < size(ink, 1) * 0.4;

keepIdx = find(keep);
if isempty(keepIdx)
    compTable = table();
    numLines = 0;
    return;
end

cleanMask = false(size(ink));
for i = 1:numel(keepIdx)
    cleanMask(CC.PixelIdxList{keepIdx(i)}) = true;
end

bands = detect_line_bands(cleanMask, bbx(keepIdx, 4));
numLines = size(bands, 1);
if numLines == 0
    compTable = table();
    return;
end

bbx = bbx(keepIdx, :);

lineIdx = assign_lines(bbx, bands);
valid = lineIdx > 0;
bbx = bbx(valid, :);
lineIdx = lineIdx(valid);

compTable = table(bbx, lineIdx, 'VariableNames', {'BoundingBox', 'Line'});
compTable.PositionInLine = zeros(height(compTable), 1);

for L = 1:numLines
    rows = find(compTable.Line == L);
    if isempty(rows)
        continue;
    end
    [~, ord] = sort(compTable.BoundingBox(rows, 1));
    compTable.PositionInLine(rows(ord)) = 1:numel(rows);
end

[~, sortIdx] = sortrows([compTable.Line, compTable.PositionInLine]);
compTable = compTable(sortIdx, :);
end

function [cleaned, strokeR] = denoise_by_stroke_width(ink)
% Estimate the page's typical stroke half-width from the mode of the
% ink's distance transform (robust to a handful of large merged blobs,
% unlike a mean/median over all ink pixels), then drop components too
% small to be a stroke fragment at that scale. strokeR (the half-width
% estimate itself) is also returned for reuse as a morphological closing
% radius by the caller.
if ~any(ink(:))
    cleaned = ink;
    strokeR = 1;
    return;
end

D = bwdist(~ink);
vals = D(ink);
vals = vals(vals > 0);
if isempty(vals)
    cleaned = ink;
    strokeR = 1;
    return;
end

edgesR = 0:1:20;
counts = histcounts(vals, edgesR);
[~, mi] = max(counts);
modeR = edgesR(mi) + 0.5;
strokeR = modeR;
strokeW = 2 * modeR;

floorArea = max(3, round(strokeW^2 * 0.4));
cleaned = bwareaopen(ink, floorArea);
end

function bands = detect_line_bands(mask, compHeights)
% Split text lines by finding valleys in the (smoothed) row-wise ink
% profile. Component-level extraction packs glyph pieces densely enough
% that the gap between two lines is rarely a true all-zero row (a
% descender or vowel sign from one line can dip into the next), so a
% simple "contiguous nonzero run" split under-counts lines badly on
% these pages -- valley-finding on the profile is far more reliable.
rowInk = sum(mask, 2);
n = numel(rowInk);
if n == 0 || isempty(compHeights)
    bands = zeros(0, 2);
    return;
end

medH = median(compHeights);
smoothed = smoothdata(double(rowInk), 'movmean', max(3, round(medH / 3)));

try
    [~, locs] = findpeaks(-smoothed, ...
        'MinPeakDistance', max(3, medH * 1.2), ...
        'MinPeakProminence', max(1, range(smoothed) * 0.15));
catch
    % Signal Processing Toolbox unavailable: fall back to splitting on
    % contiguous all-zero row runs (coarser, but dependency-free).
    hasInk = rowInk > 0;
    d = diff([0; hasInk(:); 0]);
    starts = find(d == 1);
    ends = find(d == -1) - 1;
    if isempty(starts)
        bands = zeros(0, 2);
        return;
    end
    bands = [starts, ends];
    return;
end

edgesRows = [0; locs(:); n];
bands = [edgesRows(1:end - 1) + 1, edgesRows(2:end)];

keep = (bands(:, 2) - bands(:, 1)) >= max(3, medH * 0.3);
bands = bands(keep, :);
end

function lineIdx = assign_lines(bbx, bands)
n = size(bbx, 1);
lineIdx = zeros(n, 1);
centersY = bbx(:, 2) + bbx(:, 4) / 2;
bandMid = mean(bands, 2);
for i = 1:n
    hit = find(centersY(i) >= bands(:, 1) & centersY(i) <= bands(:, 2));
    if isempty(hit)
        [~, hit] = min(abs(centersY(i) - bandMid));
    end
    lineIdx(i) = hit(1);
end
end
