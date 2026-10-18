% Which functions used by the project accept gpuArray / single inputs in this release.
I = imread('1-1-2b.jpg');
gG = gpuArray(rgb2gray(I));
bwG = gpuArray(imbinarize(rgb2gray(I), 'global'));
img = gpuArray(rand(32, 'single'));
tests = {
    'rgb2gray',    @() rgb2gray(gpuArray(I))
    'imbinarize',  @() imbinarize(gG, 'global')
    'imrotate',    @() imrotate(gG, 1.3, 'bilinear', 'loose')
    'imrotate_nn', @() imrotate(bwG, 1.3, 'nearest', 'loose')
    'edge_canny',  @() edge(bwG, 'canny')
    'hough',       @() hough(edge(bwG, 'canny'))
    'bwconncomp',  @() bwconncomp(bwG, 8)
    'regionprops', @() regionprops(bwG, 'Area', 'BoundingBox')
    'regionprops_CC', @() regionprops(bwconncomp(gather(bwG), 4), 'Area', 'Extent', 'BoundingBox')
    'bwdist',      @() bwdist(~bwG)
    'bwareaopen',  @() bwareaopen(bwG, 5)
    'imopen_line', @() imopen(bwG, strel('line', 101, 90))
    'imresize',    @() imresize(img, [20 20])
    'imresize_s',  @() imresize(img, 0.9)
    'imtranslate', @() imtranslate(img, [2 0], 'FillValues', 1)
    'imdilate4D',  @() imdilate(repmat(img, 1, 1, 1, 4), strel('disk', 1))
    'histcounts',  @() histcounts(gpuArray(rand(100, 1, 'single')), 0:0.1:1)
    'smoothdata',  @() smoothdata(gpuArray(rand(100, 1, 'single')), 'movmean', 5)
    'findpeaks_s', @() findpeaks(rand(100, 1, 'single'))
    'findpeaks_g', @() findpeaks(gpuArray(rand(100, 1, 'single')))
    'tsne_single', @() tsne(rand(200, 16, 'single'))
    'tsne_gpu',    @() tsne(gpuArray(rand(200, 16, 'single')))
    'kmeans_gpu',  @() kmeans(gpuArray(rand(200, 16, 'single')), 5)
    'eye_like',    @() eye(4, 'like', gpuArray(single(0)))
    };
for k = 1:size(tests, 1)
    try
        out = tests{k, 2}();
        if isstruct(out)
            cls = 'struct';
        else
            cls = sprintf('%s/%s', class(out), underlyingType(out));
        end
        fprintf('%-15s OK   -> %s\n', tests{k, 1}, cls);
    catch e
        fprintf('%-15s FAIL -> %s\n', tests{k, 1}, e.message);
    end
end

% trainnet with gpuArray inputs, and what precision/device its learnables end up on
X = gpuArray(rand(32, 32, 1, 64, 'single'));
net = dlnetwork([imageInputLayer([32 32 1], 'Normalization', 'none'), convolution2dLayer(3, 4, 'Padding', 'same'), ...
    convolution2dLayer(1, 1)]);
try
    tn = trainnet(X, X, net, 'mse', trainingOptions('adam', 'MaxEpochs', 1, 'Verbose', false));
    w = tn.Learnables.Value{1};
    fprintf('trainnet gpuArray input OK, learnables: %s/%s\n', class(w), underlyingType(w));
    y = predict(tn, X);
    fprintf('predict gpuArray in -> %s/%s\n', class(y), underlyingType(y));
catch e
    fprintf('trainnet gpuArray input FAIL -> %s\n', e.message);
end
