%RUN_TIBET_PIPELINE Reproduce the verified Tibetan glyph pipeline end to end.
%
%   Stages (each one was verified interactively before being added here):
%     1. Data. For cfg.dataset = 'tibetan' (default), segment pages into
%        glyph crops (segment_page.m):
%          - real woodblock scans: 1-1-2b.jpg + samples_adarshah/*.jpg
%          - clean printed transcription: printed_transcription_test.png
%        For the digit datasets, build digits/<set>/<label>/*.png once
%        if missing:
%          - 'digits_printed': 0-9 in every usable installed font
%            (make_printed_digits.m)
%          - 'digits_mnist': handwritten MNIST test set (make_mnist_digits.m)
%     2. (Optional, off by default: cfg.runFCBaseline, Tibetan only)
%        fully-connected ReLU+BN autoencoder on the scan crops, only
%        useful to re-show the conv-vs-FC invariance comparison
%     3. Convolutional autoencoder on the clean crops / digits (reconstruction only)
%     4. Contrastive (NT-Xent) fine-tuning of the conv encoder -> netCC
%     5. Shift/scale invariance measurement for every model
%     6. t-SNE of netCC features + densest-cluster montages. Digits also
%        get a t-SNE colored by true label and label-based scores:
%        leave-one-out kNN accuracy, and K-means purity / NMI (purity =
%        accuracy of labeling every cluster with its majority label).
%     7. Summary: metrics.txt, figures, models.mat in results/<timestamp>/
%
%   NOT included: alignment-based auto-labeling (blocked, wrong results)
%   and K-means label propagation (planned, not yet built/verified).
%
%   Usage:
%     run_tibet_pipeline                       % full run
%     cfgOverride = struct('quick', true);     % smoke test, a few epochs
%     run_tibet_pipeline
%     cfgOverride = struct('dataset', 'digits_printed');   % printed digits
%     run_tibet_pipeline
%     cfgOverride = struct('dataset', 'digits_mnist');     % handwritten digits
%     run_tibet_pipeline
%
%   printed_transcription_test.png must already exist: it is a screenshot
%   of the live Adarshah reader (text under folio 1-1-1b of TILO1), not
%   something this script generates.

%% 0. Configuration
cfg = struct();
cfg.dataset        = 'tibetan'; % 'tibetan' | 'digits_printed' | 'digits_mnist'
cfg.quick          = false;
cfg.maxSamples     = Inf;       % cap on images used in stages 3-6
cfg.knnK           = 5;         % digits only: neighbours for the kNN score
cfg.resegment      = true;
cfg.runFCBaseline  = false;
cfg.imgSize        = 32;
cfg.fcMaxSamples   = 4000;
cfg.fcEpochs       = 300;
cfg.convEpochs     = 150;
cfg.contrastEpochs = 60;
cfg.contrastBatch  = 64;
cfg.contrastLR     = 1e-3;
cfg.tau            = 0.1;
cfg.seed           = 1;
cfg.numClusterMontages = 4;

override = struct();
if exist('cfgOverride', 'var') && isstruct(cfgOverride)
    override = cfgOverride;
end
if isfield(override, 'dataset')
    cfg.dataset = override.dataset;
end
assert(any(strcmp(cfg.dataset, {'tibetan', 'digits_printed', 'digits_mnist'})), ...
    'Unknown cfg.dataset "%s".', cfg.dataset);
isDigits = startsWith(cfg.dataset, 'digits');
if isDigits
    % Thousands of images instead of 853: fewer AE epochs still give more
    % updates, and bigger contrastive batches give NT-Xent more negatives.
    cfg.convEpochs = 30;
    cfg.contrastBatch = 256;
end
fn = fieldnames(override);
for i = 1:numel(fn)
    cfg.(fn{i}) = override.(fn{i});
end
if cfg.quick
    cfg.fcEpochs = 3;
    cfg.convEpochs = 3;
    cfg.contrastEpochs = 2;
    cfg.fcMaxSamples = 500;
    cfg.maxSamples = min(cfg.maxSamples, 2000);
end
assert(cfg.imgSize == 32, 'The conv autoencoder decoder is built for 32x32 inputs.');

projDir = fileparts(mfilename('fullpath'));
cd(projDir);
addpath(projDir);

stamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
if isDigits
    stamp = stamp + "_" + cfg.dataset;
end
if cfg.quick
    stamp = stamp + "_quick";
end
outDir = fullfile(projDir, 'results', stamp);
mkdir(outDir);
diary(fullfile(outDir, 'run_log.txt'));

useGPU = false;
try
    useGPU = canUseGPU();
catch
end

close all;
rng(cfg.seed);
if useGPU
    gpurng(cfg.seed);
end
timings = struct();
fprintf('=== Glyph pipeline | %s | dataset=%s | quick=%d | GPU=%d ===\n', stamp, cfg.dataset, cfg.quick, useGPU);
fprintf('Results folder: %s\n', outDir);

%% 1. Data: page segmentation (Tibetan) or digit dataset preparation
t0 = tic;
switch cfg.dataset
    case 'tibetan'
        cleanImage = fullfile(projDir, 'printed_transcription_test.png');
        assert(isfile(cleanImage), ['Missing %s. It is a screenshot of the live Adarshah ' ...
            'reader (TILO1, folio 1-1-1b text) and must be copied over with the project.'], cleanImage);

        scanList = dir(fullfile(projDir, 'samples_adarshah', '*.jpg'));
        scanPages = [{fullfile(projDir, '1-1-2b.jpg')}, fullfile({scanList.folder}, {scanList.name})];
        scanOutDirs = [{fullfile(projDir, '1-1-2b_crops')}, ...
            cellfun(@(n) fullfile(projDir, 'samples_adarshah_crops', erase(n, '.jpg')), {scanList.name}, 'UniformOutput', false)];
        cleanOutDir = fullfile(projDir, 'printed_transcription_crops');

        if cfg.resegment
            % Wipe old crops first: crop filenames encode line/position, so a run
            % that finds a different count would otherwise leave stale files behind.
            for d = [scanOutDirs, {cleanOutDir}]
                if isfolder(d{1})
                    rmdir(d{1}, 's');
                end
            end
            nScanCrops = 0;
            for i = 1:numel(scanPages)
                r = segment_page(scanPages{i}, 'OutputDir', scanOutDirs{i});
                fprintf('  %-40s %5d crops\n', scanPages{i}(numel(projDir)+2:end), numel(r));
                nScanCrops = nScanCrops + numel(r);
            end
            rClean = segment_page(cleanImage, 'OutputDir', cleanOutDir);
            fprintf('  %-40s %5d crops, %d lines\n', 'printed_transcription_test.png', numel(rClean), max([rClean.Line]));
            fprintf('  total scan crops: %d\n', nScanCrops);
        else
            assert(all(cellfun(@isfolder, [scanOutDirs, {cleanOutDir}])), ...
                'cfg.resegment=false but crop folders are missing; run with resegment=true once.');
            fprintf('  reusing existing crop folders\n');
        end
        dataDir = cleanOutDir;
        dataTitle = 'Clean printed Tibetan text';
    case 'digits_printed'
        dataDir = fullfile(projDir, 'digits', 'printed');
        if ~isfolder(dataDir)
            make_printed_digits(dataDir);
        end
        dataTitle = 'Printed digits (installed fonts)';
    case 'digits_mnist'
        dataDir = fullfile(projDir, 'digits', 'mnist');
        if ~isfolder(dataDir)
            make_mnist_digits(dataDir, fullfile(projDir, 'digits', 'raw'), "test");
        end
        dataTitle = 'Handwritten digits (MNIST test set)';
end
timings.data = toc(t0);

%% 2. Baseline: fully-connected ReLU+BN autoencoder on scan crops
sz = cfg.imgSize;
netFC = [];
if cfg.runFCBaseline && strcmp(cfg.dataset, 'tibetan')
    t0 = tic;
    [Xn, ~] = load_crop_dataset({fullfile(projDir, '1-1-2b_crops'), fullfile(projDir, 'samples_adarshah_crops')}, sz, cfg.fcMaxSamples);
    Xn = Xn';
    if useGPU
        Xn = gpuArray(Xn);
    end
    fcOpts = trainingOptions('adam', ...
        'MaxEpochs', cfg.fcEpochs, 'MiniBatchSize', 256, 'InitialLearnRate', 5e-3, ...
        'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', 100, 'LearnRateDropFactor', 0.5, ...
        'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none');
    netFC = trainnet(Xn, Xn, build_fc_autoencoder(sz * sz), "mse", fcOpts);
    mseFC = gather(mean((predict(netFC, Xn) - Xn).^2, 'all'));
    timings.fcBaseline = toc(t0);
    fprintf('Stage 2: FC baseline trained on %d scan crops, recon MSE=%.5f (%.0fs)\n', size(Xn, 1), mseFC, timings.fcBaseline);
end

%% 3. Convolutional autoencoder on clean crops / digits
t0 = tic;
[Xp, metaP] = load_crop_dataset({dataDir}, sz, cfg.maxSamples);
labels = [];
if isDigits
    labels = folder_labels(metaP);
end
imgs = reshape(Xp, sz, sz, 1, []);
if useGPU
    imgs = gpuArray(imgs);
end
convOpts = trainingOptions('adam', ...
    'MaxEpochs', cfg.convEpochs, 'MiniBatchSize', 64, 'InitialLearnRate', 2e-3, ...
    'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none');
[netConvT, infoConv] = trainnet(imgs, imgs, build_conv_autoencoder(sz), "mse", convOpts);
mseConv = gather(mean((predict(netConvT, imgs) - imgs).^2, 'all'));
timings.convAE = toc(t0);
fprintf('Stage 3: conv AE trained on %d images (%s), recon MSE=%.5f (%.0fs)\n', size(imgs, 4), dataTitle, mseConv, timings.convAE);

figure(1);
hConv = infoConv.TrainingHistory;
plot(hConv.Iteration, hConv.Loss, 'Color', [0.7 0.7 0.7]); hold on;
plot(hConv.Iteration, movmean(hConv.Loss, 25), 'r-', 'LineWidth', 2); hold off;
xlabel('Iteration'); ylabel('MSE'); grid on;
title('Conv autoencoder reconstruction loss');
exportgraphics(figure(1), fullfile(outDir, 'fig1_conv_ae_loss.png'), 'Resolution', 150);

%% 4. Contrastive fine-tuning of the conv encoder
t0 = tic;
bottleneckIdx = find(strcmp({netConvT.Layers.Name}, 'bottleneck'));
projHead = [
    fullyConnectedLayer(32, 'Name', 'proj1')
    reluLayer('Name', 'proj_relu')
    fullyConnectedLayer(16, 'Name', 'proj2')
];
netCC = dlnetwork([netConvT.Layers(1:bottleneckIdx); projHead]);
if useGPU
    netCC = to_device(netCC, @gpuArray);
end

nSamples = size(imgs, 4);
numBatches = floor(nSamples / cfg.contrastBatch);
avgGrad = [];
avgSqGrad = [];
iteration = 0;
contrastLoss = zeros(cfg.contrastEpochs, 1, 'single');
for epoch = 1:cfg.contrastEpochs
    order = randperm(nSamples);
    % Kept on the device so logging the loss doesn't force a GPU sync every iteration.
    batchLoss = zeros(numBatches, 1, 'like', imgs);
    for b = 1:numBatches
        idx = order((b - 1) * cfg.contrastBatch + 1 : b * cfg.contrastBatch);
        dlXb = dlarray(make_contrastive_batch_img(imgs, idx), 'SSCB');
        iteration = iteration + 1;
        [loss, gradients, state] = dlfeval(@contrastive_model_loss, netCC, dlXb, cfg.tau);
        netCC.State = state;
        [netCC, avgGrad, avgSqGrad] = adamupdate(netCC, gradients, avgGrad, avgSqGrad, iteration, cfg.contrastLR);
        batchLoss(b) = extractdata(loss);
    end
    contrastLoss(epoch) = gather(mean(batchLoss));
    if epoch == 1 || mod(epoch, 10) == 0 || epoch == cfg.contrastEpochs
        fprintf('  contrastive epoch %3d/%d  loss=%.4f  (%.0fs)\n', epoch, cfg.contrastEpochs, contrastLoss(epoch), toc(t0));
    end
end
timings.contrastive = toc(t0);
fprintf('Stage 4: contrastive fine-tuning done, final loss=%.4f (%.0fs)\n', contrastLoss(end), timings.contrastive);

figure(2);
plot(1:cfg.contrastEpochs, contrastLoss, 'o-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('mean NT-Xent loss'); grid on;
title('Contrastive fine-tuning loss (conv encoder)');
exportgraphics(figure(2), fullfile(outDir, 'fig2_contrastive_loss.png'), 'Resolution', 150);

%% 5. Shift/scale invariance of each model
t0 = tic;
% Encoders map an sz x sz x 1 x B image batch to D x B features.
encodeConv = @(net) @(X) squeeze(predict(net, X, 'Outputs', 'bottleneck'));
models = {'Conv AE (recon only)', encodeConv(netConvT); 'Conv + contrastive (netCC)', encodeConv(netCC)};
if ~isempty(netFC)
    encodeFC = @(X) predict(netFC, reshape(X, sz * sz, [])', 'Outputs', 'bottleneck')';
    models = [{'FC ReLU+BN (baseline)', encodeFC}; models];
end
invariance = struct('model', {}, 'shiftMean', {}, 'shiftWorst', {}, 'scaleMean', {}, 'scaleWorst', {});
for m = 1:size(models, 1)
    inv = measure_invariance(models{m, 2}, imgs);
    inv.model = models{m, 1};
    invariance(end + 1) = inv; %#ok<SAGROW>
end
timings.invariance = toc(t0);
fprintf('Stage 5: invariance (cosine sim to unperturbed glyph; mean / worst-case)\n');
for m = 1:numel(invariance)
    fprintf('  %-28s shift %.3f / %.3f   scale %.3f / %.3f\n', invariance(m).model, ...
        invariance(m).shiftMean, invariance(m).shiftWorst, invariance(m).scaleMean, invariance(m).scaleWorst);
end

%% 6. t-SNE of netCC features + densest-cluster montages
t0 = tic;
% tsne has no gpuArray support; it runs on the CPU in single.
fCC = gather(squeeze(predict(netCC, imgs, 'Outputs', 'bottleneck')));
rng(0);
Ycc = tsne(fCC', 'Perplexity', 30);
render_tsne_canvas(Ycc, metaP, struct('FigNum', 3, 'Title', [dataTitle ': conv encoder + contrastive (netCC)']));
exportgraphics(figure(3), fullfile(outDir, 'fig3_tsne_netCC.png'), 'Resolution', 200);

regions = find_dense_regions(Ycc, cfg.numClusterMontages);
for r = 1:numel(regions)
    members = regions{r}(1:min(16, numel(regions{r})));
    thumbs = cellfun(@(p) imresize(imread(p), [70 70]), {metaP(members).Path}, 'UniformOutput', false);
    figNum = 3 + r;
    figure(figNum);
    montage(thumbs, 'BorderSize', [3 3], 'BackgroundColor', 'white');
    title(sprintf('Densest t-SNE cluster #%d (%d members, %d shown)', r, numel(regions{r}), numel(members)));
    exportgraphics(figure(figNum), fullfile(outDir, sprintf('fig%d_cluster%d.png', figNum, r)), 'Resolution', 150);
end
timings.tsne = toc(t0);
fprintf('Stage 6: t-SNE + %d cluster montages (%.0fs)\n', numel(regions), timings.tsne);

labelMetrics = struct('features', {}, 'knnAcc', {}, 'kmeansPurity', {}, 'kmeansNMI', {});
if isDigits
    t0 = tic;
    figNum = 4 + numel(regions);
    figure(figNum);
    gscatter(Ycc(:, 1), Ycc(:, 2), labels, turbo(10), '.', 6);
    legend('Location', 'eastoutside');
    axis equal off;
    title([dataTitle ': t-SNE of netCC features, colored by true digit']);
    exportgraphics(figure(figNum), fullfile(outDir, sprintf('fig%d_tsne_true_labels.png', figNum)), 'Resolution', 150);

    % Raw pixels are taken ink-positive so cosine similarity isn't dominated by the shared white background.
    featSets = {
        'Raw pixels (baseline)', reshape(1 - imgs, sz * sz, [])
        'Conv AE (recon only)', squeeze(predict(netConvT, imgs, 'Outputs', 'bottleneck'))
        'Conv + contrastive (netCC)', squeeze(predict(netCC, imgs, 'Outputs', 'bottleneck'))
        };
    for m = 1:size(featSets, 1)
        [acc, pur, nmi] = label_metrics(featSets{m, 2}, labels, cfg.knnK, 10);
        labelMetrics(end + 1) = struct('features', featSets{m, 1}, 'knnAcc', acc, ... %#ok<SAGROW>
            'kmeansPurity', pur, 'kmeansNMI', nmi);
    end
    timings.labelMetrics = toc(t0);
    fprintf('Stage 6b: label scores (%d-NN leave-one-out accuracy | K-means k=10 purity / NMI) (%.0fs)\n', ...
        cfg.knnK, timings.labelMetrics);
    for m = 1:numel(labelMetrics)
        fprintf('  %-28s kNN %.3f   purity %.3f   NMI %.3f\n', labelMetrics(m).features, ...
            labelMetrics(m).knnAcc, labelMetrics(m).kmeansPurity, labelMetrics(m).kmeansNMI);
    end
end

%% 7. Summary
% Gathered so models.mat also loads on a machine without a GPU.
netCC = to_device(netCC, @gather);
save(fullfile(outDir, 'models.mat'), 'netFC', 'netConvT', 'netCC', 'fCC', 'Ycc', 'metaP', 'labels', ...
    'invariance', 'labelMetrics', 'contrastLoss', 'cfg', 'timings', '-v7.3');

fid = fopen(fullfile(outDir, 'metrics.txt'), 'w');
fprintf(fid, 'Glyph pipeline run %s, dataset %s (quick=%d, GPU=%d)\n\n', stamp, cfg.dataset, cfg.quick, useGPU);
fprintf(fid, '%s: %d images   Conv AE recon MSE: %.5f\n', dataTitle, size(imgs, 4), mseConv);
if ~isempty(netFC)
    fprintf(fid, 'FC baseline recon MSE: %.5f\n', mseFC);
end
fprintf(fid, 'Contrastive final loss: %.4f\n\n', contrastLoss(end));
fprintf(fid, 'Invariance (cosine similarity to unperturbed glyph)\n');
fprintf(fid, '%-28s %10s %10s %10s %10s\n', 'model', 'shiftMean', 'shiftWorst', 'scaleMean', 'scaleWorst');
for m = 1:numel(invariance)
    fprintf(fid, '%-28s %10.3f %10.3f %10.3f %10.3f\n', invariance(m).model, ...
        invariance(m).shiftMean, invariance(m).shiftWorst, invariance(m).scaleMean, invariance(m).scaleWorst);
end
if ~isempty(labelMetrics)
    fprintf(fid, '\nLabel scores: %d-NN leave-one-out accuracy (cosine), K-means k=10 purity and NMI\n', cfg.knnK);
    fprintf(fid, '%-28s %10s %10s %10s\n', 'features', 'kNN', 'purity', 'NMI');
    for m = 1:numel(labelMetrics)
        fprintf(fid, '%-28s %10.3f %10.3f %10.3f\n', labelMetrics(m).features, ...
            labelMetrics(m).knnAcc, labelMetrics(m).kmeansPurity, labelMetrics(m).kmeansNMI);
    end
end
fprintf(fid, '\nStage timings (s):\n');
tf = fieldnames(timings);
for i = 1:numel(tf)
    fprintf(fid, '  %-14s %8.1f\n', tf{i}, timings.(tf{i}));
end
fclose(fid);

fprintf('=== Done. Metrics, figures, models.mat and run_log.txt are in:\n    %s\n', outDir);
diary off;

%% Local functions
function net = to_device(net, fn)
% Move a dlnetwork's learnables and state (e.g. batchnorm statistics),
% which dlupdate on the network alone leaves behind, with fn = @gpuArray
% or @gather.
net = dlupdate(fn, net);
net.State = dlupdate(fn, net.State);
end

function labels = folder_labels(meta)
% Digit datasets are stored as <root>/<label>/<file>.png.
[~, names] = cellfun(@(p) fileparts(fileparts(p)), {meta.Path}, 'UniformOutput', false);
labels = single(str2double(names))';
assert(~any(isnan(labels)), 'Could not parse a digit label from every parent folder name.');
end

function net = build_fc_autoencoder(nIn)
layers = [
    featureInputLayer(nIn, 'Name', 'input')
    fullyConnectedLayer(256, 'Name', 'enc1', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(128, 'Name', 'enc2', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(32, 'Name', 'enc3', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'bottleneck')
    fullyConnectedLayer(128, 'Name', 'dec3', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    fullyConnectedLayer(256, 'Name', 'dec2', 'WeightsInitializer', 'he')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    fullyConnectedLayer(nIn, 'Name', 'dec1', 'WeightsInitializer', 'he')
    sigmoidLayer('Name', 'reconstruction')
];
net = dlnetwork(layers);
end

function net = build_conv_autoencoder(sz)
layers = [
    imageInputLayer([sz sz 1], 'Normalization', 'none', 'Name', 'input')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    globalAveragePooling2dLayer('Name', 'bottleneck')
    transposedConv2dLayer(4, 32, 'Stride', 4, 'Name', 'tconv1')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    transposedConv2dLayer(2, 32, 'Stride', 2, 'Name', 'tconv2')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    transposedConv2dLayer(2, 16, 'Stride', 2, 'Name', 'tconv3')
    batchNormalizationLayer('Name', 'bn6')
    reluLayer('Name', 'relu6')
    transposedConv2dLayer(2, 1, 'Stride', 2, 'Name', 'tconv4')
    sigmoidLayer('Name', 'reconstruction')
];
net = dlnetwork(layers);
end

function inv = measure_invariance(encodeFn, imgs)
% Same protocol as the interactive runs: 8 fixed clean glyphs, +/-1..3 px
% horizontal shifts and 0.85-1.15x scaling, cosine similarity of each
% perturbed embedding to the unperturbed one. All perturbed copies are
% built as one batch and encoded in a single call.
rng(7);
testIdx = randperm(size(imgs, 4), 8);
shifts = [-3 -2 -1 1 2 3];
scales = [0.85 0.9 0.95 1.05 1.1 1.15];
sz = size(imgs, 1);
base = imgs(:, :, :, testIdx);

pert = cell(1, numel(shifts) + numel(scales));
for k = 1:numel(shifts)
    pert{k} = shift_columns(base, shifts(k));
end
for k = 1:numel(scales)
    pert{numel(shifts) + k} = center_on_canvas(imresize(base, scales(k)), sz);
end

f0 = encodeFn(base);
F = reshape(encodeFn(cat(4, pert{:})), size(f0, 1), numel(testIdx), []);
sims = gather(sum(f0 .* F, 1) ./ (vecnorm(f0, 2, 1) .* vecnorm(F, 2, 1) + eps('single')));
simShift = sims(:, :, 1:numel(shifts));
simScale = sims(:, :, numel(shifts) + 1:end);
inv = struct('model', '', 'shiftMean', mean(simShift, 'all'), 'shiftWorst', min(simShift, [], 'all'), ...
    'scaleMean', mean(simScale, 'all'), 'scaleWorst', min(simScale, [], 'all'));
end

function out = shift_columns(img, s)
% Integer horizontal shift with white fill, same as imtranslate(img, [s 0], 'FillValues', 1).
out = ones(size(img), 'like', img);
if s > 0
    out(:, s + 1:end, :, :) = img(:, 1:end - s, :, :);
else
    out(:, 1:end + s, :, :) = img(:, 1 - s:end, :, :);
end
end

function canvas = center_on_canvas(img, sz)
[h, w, c, n] = size(img);
canvas = ones(sz, sz, c, n, 'like', img);
r0 = max(1, floor((sz - h) / 2) + 1);
c0 = max(1, floor((sz - w) / 2) + 1);
rEnd = min(sz, r0 + h - 1);
cEnd = min(sz, c0 + w - 1);
canvas(r0:rEnd, c0:cEnd, :, :) = img(1:(rEnd - r0 + 1), 1:(cEnd - c0 + 1), :, :);
end

function regions = find_dense_regions(Y, nRegions)
% Greedy grid scan over the normalized t-SNE plane for the most populated
% boxes, skipping boxes that overlap one already picked.
Y = single(Y);
Yn = (Y - min(Y, [], 1)) ./ max(max(Y, [], 1) - min(Y, [], 1), eps('single'));
boxSize = single(0.065);
step = single(0.04);
starts = 0:step:(1 - boxSize);
[CX, CY] = meshgrid(starts, starts); % column-major order = cx outer, cy inner
CX = CX(:)';
CY = CY(:)';
inBox = Yn(:, 1) >= CX & Yn(:, 1) < CX + boxSize & Yn(:, 2) >= CY & Yn(:, 2) < CY + boxSize;
candidates = [CX; CY; sum(single(inBox), 1)]';
candidates = sortrows(candidates, -3);
regions = {};
picked = zeros(0, 2, 'single');
for k = 1:size(candidates, 1)
    if numel(regions) >= nRegions || candidates(k, 3) == 0
        break;
    end
    c = candidates(k, 1:2);
    if ~isempty(picked) && any(all(abs(picked - c) < boxSize, 2))
        continue;
    end
    picked(end + 1, :) = c; %#ok<AGROW>
    regions{end + 1} = find(Yn(:, 1) >= c(1) & Yn(:, 1) < c(1) + boxSize & ...
        Yn(:, 2) >= c(2) & Yn(:, 2) < c(2) + boxSize); %#ok<AGROW>
end
end
