function info = make_printed_digits(outDir, varargin)
%MAKE_PRINTED_DIGITS Render digits 0-9 in every usable installed TrueType
%font and save them as MNIST-style 32x32 crops.
%
%   INFO = MAKE_PRINTED_DIGITS(OUTDIR) writes OUTDIR/<digit>/<font>.png
%   and returns a struct with the accepted and rejected font names.
%
%   Each digit is normalized the way MNIST was built: the ink bounding box
%   is scaled so its longer side is 20 px (aspect ratio preserved) and
%   placed in a 28x28 field with its center of mass at the center. That
%   field is then padded by 2 px to 32x32, which is what the conv
%   autoencoder expects and also what make_mnist_digits produces, so the
%   printed and handwritten sets are directly comparable. Images are saved
%   with a white background and black ink, the convention of the glyph
%   crops and load_crop_dataset.
%
%   A font is rejected when it is a symbol/icon font (by name), lacks a
%   digit glyph or fails to render, renders a digit with no ink or
%   clipped at the canvas edge, or renders two
%   different digits identically (missing glyphs all drawn as the same
%   placeholder box). Fonts whose ten digits are pixel-identical to an
%   already accepted font (e.g. "Microsoft YaHei" / "Microsoft YaHei UI")
%   are skipped as duplicates.
%
%   Name-value options:
%       'FontSize' - render size before normalization (default 96)

p = inputParser;
addParameter(p, 'FontSize', 96, @(x) isnumeric(x) && isscalar(x));
parse(p, varargin{:});
fontSize = p.Results.FontSize;

symbolFonts = ["Webdings", "Wingdings", "Symbol", "Monotype Sorts", "Holidays MT", ...
    "Segoe Fluent Icons", "Segoe MDL2 Assets", "Segoe UI Emoji"];
fonts = string(listTrueTypeFonts());
isSymbol = startsWith(fonts, symbolFonts);

canvasSize = 4 * fontSize;
digits = '0123456789';
% A font without a digit glyph should be rejected, not silently rendered
% with a fallback, so make insertText's warning catchable.
warnState = warning('error', 'vision:insertText:missingCharInFont'); %#ok<CTPCT>
restoreWarn = onCleanup(@() warning(warnState));
accepted = strings(0, 1);
rejected = table(fonts(isSymbol), repmat("symbol font", nnz(isSymbol), 1), ...
    'VariableNames', {'Font', 'Reason'});
kept = zeros(32, 32, 10, 0, 'single');

for f = fonts(~isSymbol)'
    glyphs = zeros(32, 32, 10, 'single');
    reason = "";
    for d = 1:10
        try
            rgb = insertText(ones(canvasSize, canvasSize, 'single'), [canvasSize canvasSize] / 2, digits(d), ...
                'Font', char(f), 'FontSize', fontSize, 'BoxOpacity', 0, 'TextColor', 'black', ...
                'AnchorPoint', 'Center');
        catch err
            reason = "render failed for '" + digits(d) + "': " + err.message;
            break;
        end
        ink = 1 - rgb2gray(rgb);
        if ~any(ink > 0.05, 'all')
            reason = "no ink for '" + digits(d) + "'";
            break;
        end
        if any(ink([1 end], :) > 0.05, 'all') || any(ink(:, [1 end]) > 0.05, 'all')
            reason = "'" + digits(d) + "' clipped at canvas edge";
            break;
        end
        glyphs(:, :, d) = mnist_normalize(ink);
    end
    if reason == ""
        flat = reshape(glyphs, [], 10);
        [~, firstIdx] = unique(flat', 'rows', 'stable');
        if numel(firstIdx) < 10
            reason = "identical renders for different digits (missing glyphs)";
        elseif any(arrayfun(@(k) isequal(kept(:, :, :, k), glyphs), 1:size(kept, 4)))
            reason = "duplicate of an accepted font";
        end
    end
    if reason ~= ""
        rejected(end + 1, :) = {f, reason}; %#ok<AGROW>
        continue;
    end
    kept(:, :, :, end + 1) = glyphs; %#ok<AGROW>
    accepted(end + 1) = f; %#ok<AGROW>
end

for d = 1:10
    mkdir(fullfile(outDir, digits(d)));
end
for k = 1:numel(accepted)
    name = regexprep(accepted(k), '[^A-Za-z0-9]+', '_');
    for d = 1:10
        imwrite(im2uint8(1 - kept(:, :, d, k)), fullfile(outDir, digits(d), sprintf('%03d_%s.png', k, name)));
    end
end

info = struct('accepted', accepted, 'rejected', rejected);
fprintf('make_printed_digits: %d fonts accepted (%d images), %d rejected -> %s\n', ...
    numel(accepted), 10 * numel(accepted), height(rejected), outDir);
end

function out = mnist_normalize(ink)
% Ink-positive image in [0,1] -> 32x32 ink-positive, MNIST-style framing.
rows = find(any(ink > 0.05, 2));
cols = find(any(ink > 0.05, 1));
patch = ink(rows(1):rows(end), cols(1):cols(end));
[h, w] = size(patch);
patch = imresize(patch, round([h w] * 20 / max(h, w)), 'bilinear', 'Antialiasing', true);
patch = max(0, min(1, patch));

[ph, pw] = size(patch);
[cc, rr] = meshgrid(cast(1:pw, 'like', patch), cast(1:ph, 'like', patch));
mass = sum(patch, 'all');
comR = sum(rr .* patch, 'all') / mass;
comC = sum(cc .* patch, 'all') / mass;

% Row r0 + comR - 1 should land on the field center 14.5.
r0 = min(max(round(15.5 - comR), 1), 28 - ph + 1);
c0 = min(max(round(15.5 - comC), 1), 28 - pw + 1);
field = zeros(28, 28, 'like', patch);
field(r0:r0 + ph - 1, c0:c0 + pw - 1) = patch;
out = zeros(32, 32, 'like', patch);
out(3:30, 3:30) = field;
end
