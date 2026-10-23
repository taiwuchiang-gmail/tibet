function counts = make_mnist_digits(outDir, rawDir, split)
%MAKE_MNIST_DIGITS Download MNIST and save it as 32x32 crops.
%
%   COUNTS = MAKE_MNIST_DIGITS(OUTDIR, RAWDIR, SPLIT) downloads the MNIST
%   SPLIT ("test", 10,000 images, default; or "train", 60,000) into
%   RAWDIR (skipped if already there) and writes
%   OUTDIR/<digit>/mnist_<split>_<index>.png. COUNTS(d+1) is the number
%   of images of digit d.
%
%   MNIST digits are already normalized (20x20 box, centered by center of
%   mass in a 28x28 field). They are padded by 2 px to the 32x32 the conv
%   autoencoder expects and inverted to white background / black ink, the
%   convention of the glyph crops, load_crop_dataset and
%   make_printed_digits.

if nargin < 3
    split = "test";
end
prefix = struct('test', "t10k", 'train', "train");
prefix = prefix.(char(split));
baseUrl = "https://ossci-datasets.s3.amazonaws.com/mnist/";

if ~isfolder(rawDir)
    mkdir(rawDir);
end
imgFile = fetch(baseUrl, rawDir, prefix + "-images-idx3-ubyte");
lblFile = fetch(baseUrl, rawDir, prefix + "-labels-idx1-ubyte");

images = read_idx(imgFile); % 28 x 28 x N uint8, ink = 255
labels = read_idx(lblFile);
assert(size(images, 3) == numel(labels), 'MNIST image/label count mismatch.');

for d = 0:9
    mkdir(fullfile(outDir, num2str(d)));
end
padded = 255 * ones(32, 32, 'uint8');
for i = 1:numel(labels)
    padded(3:30, 3:30) = 255 - images(:, :, i);
    imwrite(padded, fullfile(outDir, num2str(labels(i)), sprintf('mnist_%s_%05d.png', split, i)));
end

counts = accumarray(single(labels(:)) + 1, single(1), [10 1])';
fprintf('make_mnist_digits: %d %s images -> %s\n', numel(labels), split, outDir);
end

function file = fetch(baseUrl, rawDir, name)
file = fullfile(rawDir, name);
if isfile(file)
    return;
end
gz = file + ".gz";
websave(gz, baseUrl + name + ".gz");
gunzip(gz, rawDir);
delete(gz);
end

function data = read_idx(file)
% IDX format: big-endian int32 magic (last byte = number of dims), int32
% dim sizes, then row-major uint8 data.
fid = fopen(file, 'r', 'ieee-be');
cleanup = onCleanup(@() fclose(fid));
magic = fread(fid, 1, 'int32=>int32');
dims = fread(fid, mod(magic, 256), 'int32=>int32')';
data = fread(fid, Inf, 'uint8=>uint8');
if numel(dims) == 3
    data = permute(reshape(data, dims(3), dims(2), dims(1)), [2 1 3]);
end
end
