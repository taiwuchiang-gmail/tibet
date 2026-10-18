%RUN_TIBET_PIPELINE Reproduce the verified Tibetan glyph pipeline end to end.
%
%   Stages (each one was verified interactively before being added here):
%     1. Segment pages into glyph crops (segment_page.m)
%          - real woodblock scans: 1-1-2b.jpg + samples_adarshah/*.jpg
%          - clean printed transcription: printed_transcription_test.png
%     2. (Optional, off by default: cfg.runFCBaseline) fully-connected
%        ReLU+BN autoencoder on the scan crops, only useful to re-show
%        the conv-vs-FC invariance comparison
%     3. Convolutional autoencoder on the clean crops (reconstruction only)
%     4. Contrastive (NT-Xent) fine-tuning of the conv encoder -> netCC
%     5. Shift/scale invariance measurement for every model
%     6. t-SNE of netCC features on clean crops + densest-cluster montages
%     7. Summary: metrics.txt, figures, models.mat in results/<timestamp>/
%
%   NOT included: alignment-based auto-labeling (blocked, wrong results)
%   and K-means label propagation (planned, not yet built/verified).
%
%   Usage:
%     run_tibet_pipeline                       % full run
%     cfgOverride = struct('quick', true);     % smoke test, a few epochs
%     run_tibet_pipeline
%
%   printed_transcription_test.png must already exist: it is a screenshot
%   of the live Adarshah reader (text under folio 1-1-1b of TILO1), not
%   something this script generates.

%% 0. Configuration
cfg = struct();
cfg.quick          = false;
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

if exist('cfgOverride', 'var') && isstruct(cfgOverride)
    fn = fieldnames(cfgOverride);
    for i = 1:numel(fn)
        cfg.(fn{i}) = cfgOverride.(fn{i});
    end
end
if cfg.quick
    cfg.fcEpochs = 3;
    cfg.convEpochs = 3;
    cfg.contrastEpochs = 2;
    cfg.fcMaxSamples = 500;
end
assert(cfg.imgSize == 32, 'The conv autoencoder decoder is built for 32x32 inputs.');

projDir = fileparts(mfilename('fullpath'));
cd(projDir);
addpath(projDir);

stamp = string(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
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
timings = struct();
fprintf('=== Tibetan glyph pipeline | %s | quick=%d | GPU=%d ===\n', stamp, cfg.quick, useGPU);
fprintf('Results folder: %s\n', outDir);

%% 1. Segmentation
t0 = tic;
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
timings.segmentation = toc(t0);

%% 2. Baseline: fully-connected ReLU+BN autoencoder on scan crops
sz = cfg.imgSize;
netFC = [];
if cfg.runFCBaseline
    t0 = tic;
    [Xn, ~] = load_crop_dataset({fullfile(projDir, '1-1-2b_crops'), fullfile(projDir, 'samples_adarshah_crops')}, sz, cfg.fcMaxSamples);
    fcOpts = trainingOptions('adam', ...
        'MaxEpochs', cfg.fcEpochs, 'MiniBatchSize', 256, 'InitialLearnRate', 5e-3, ...
        'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', 100, 'LearnRateDropFactor', 0.5, ...
        'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none');
    netFC = trainnet(Xn', Xn', build_fc_autoencoder(sz * sz), "mse", fcOpts);
    recFC = to_numeric(predict(netFC, Xn'));
    mseFC = mean((recFC(:) - double(Xn(:))).^2, 'all');
    timings.fcBaseline = toc(t0);
    fprintf('Stage 2: FC baseline trained on %d scan crops, recon MSE=%.5f (%.0fs)\n', size(Xn, 2), mseFC, timings.fcBaseline);
end

%% 3. Convolutional autoencoder on clean crops
t0 = tic;
[Xp, metaP] = load_crop_dataset({cleanOutDir}, sz, Inf);
imgs = single(reshape(Xp, sz, sz, 1, []));
convOpts = trainingOptions('adam', ...
    'MaxEpochs', cfg.convEpochs, 'MiniBatchSize', 64, 'InitialLearnRate', 2e-3, ...
    'Shuffle', 'every-epoch', 'Verbose', false, 'Plots', 'none');
[netConvT, infoConv] = trainnet(imgs, imgs, build_conv_autoencoder(sz), "mse", convOpts);
recConv = to_numeric(predict(netConvT, imgs));
mseConv = mean((recConv(:) - double(imgs(:))).^2, 'all');
timings.convAE = toc(t0);
fprintf('Stage 3: conv AE trained on %d clean crops, recon MSE=%.5f (%.0fs)\n', size(imgs, 4), mseConv, timings.convAE);

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

nSamples = size(imgs, 4);
numBatches = floor(nSamples / cfg.contrastBatch);
avgGrad = [];
avgSqGrad = [];
iteration = 0;
contrastLoss = zeros(cfg.contrastEpochs, 1);
for epoch = 1:cfg.contrastEpochs
    order = randperm(nSamples);
    batchLoss = zeros(numBatches, 1);
    for b = 1:numBatches
        idx = order((b - 1) * cfg.contrastBatch + 1 : b * cfg.contrastBatch);
        dlXb = dlarray(make_contrastive_batch_img(imgs, idx), 'SSCB');
        if useGPU
            dlXb = gpuArray(dlXb);
        end
        iteration = iteration + 1;
        [loss, gradients, state] = dlfeval(@contrastive_model_loss, netCC, dlXb, cfg.tau);
        netCC.State = state;
        [netCC, avgGrad, avgSqGrad] = adamupdate(netCC, gradients, avgGrad, avgSqGrad, iteration, cfg.contrastLR);
        batchLoss(b) = to_numeric(loss);
    end
    contrastLoss(epoch) = mean(batchLoss);
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
encodeConv = @(net) @(im) to_numeric(predict(net, single(reshape(im, sz, sz, 1, 1)), 'Outputs', 'bottleneck'));
models = {'Conv AE (recon only)', encodeConv(netConvT); 'Conv + contrastive (netCC)', encodeConv(netCC)};
if ~isempty(netFC)
    encodeFC = @(im) to_numeric(predict(netFC, single(im(:))', 'Outputs', 'bottleneck'));
    models = [{'FC ReLU+BN (baseline)', encodeFC}; models];
end
invariance = struct('model', {}, 'shiftMean', {}, 'shiftWorst', {}, 'scaleMean', {}, 'scaleWorst', {});
for m = 1:size(models, 1)
    inv = measure_invariance(models{m, 2}, Xp, sz);
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
fCC = squeeze(to_numeric(predict(netCC, imgs, 'Outputs', 'bottleneck')));
rng(0);
Ycc = tsne(fCC', 'Perplexity', 30);
render_tsne_canvas(Ycc, metaP, struct('FigNum', 3, 'Title', 'Clean printed text: conv encoder + contrastive (netCC)'));
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

%% 7. Summary
save(fullfile(outDir, 'models.mat'), 'netFC', 'netConvT', 'netCC', 'fCC', 'Ycc', 'metaP', ...
    'invariance', 'contrastLoss', 'cfg', 'timings', '-v7.3');

fid = fopen(fullfile(outDir, 'metrics.txt'), 'w');
fprintf(fid, 'Tibetan glyph pipeline run %s (quick=%d, GPU=%d)\n\n', stamp, cfg.quick, useGPU);
fprintf(fid, 'Clean crops: %d   Conv AE recon MSE: %.5f\n', size(imgs, 4), mseConv);
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
fprintf(fid, '\nStage timings (s):\n');
tf = fieldnames(timings);
for i = 1:numel(tf)
    fprintf(fid, '  %-14s %8.1f\n', tf{i}, timings.(tf{i}));
end
fclose(fid);

fprintf('=== Done. Metrics, figures, models.mat and run_log.txt are in:\n    %s\n', outDir);
diary off;

%% Local functions
function x = to_numeric(x)
if isa(x, 'dlarray')
    x = extractdata(x);
end
x = double(gather(x));
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

function inv = measure_invariance(encodeFn, Xp, sz)
% Same protocol as the interactive runs: 8 fixed clean glyphs, +/-1..3 px
% horizontal shifts and 0.85-1.15x scaling, cosine similarity of each
% perturbed embedding to the unperturbed one.
rng(7);
testIdx = randperm(size(Xp, 2), 8);
shifts = [-3 -2 -1 1 2 3];
scales = [0.85 0.9 0.95 1.05 1.1 1.15];
simShift = [];
simScale = [];
for t = testIdx
    img0 = reshape(double(Xp(:, t)), sz, sz);
    f0 = encodeFn(img0);
    f0 = f0(:);
    for s = shifts
        f = encodeFn(imtranslate(img0, [s 0], 'FillValues', 1));
        simShift(end + 1) = cosine_sim(f0, f(:)); %#ok<AGROW>
    end
    for sc = scales
        f = encodeFn(center_on_canvas(imresize(img0, sc), sz));
        simScale(end + 1) = cosine_sim(f0, f(:)); %#ok<AGROW>
    end
end
inv = struct('model', '', 'shiftMean', mean(simShift), 'shiftWorst', min(simShift), ...
    'scaleMean', mean(simScale), 'scaleWorst', min(simScale));
end

function c = cosine_sim(a, b)
c = dot(a, b) / (norm(a) * norm(b) + eps);
end

function canvas = center_on_canvas(img, sz)
canvas = ones(sz, sz);
[h, w] = size(img);
r0 = max(1, floor((sz - h) / 2) + 1);
c0 = max(1, floor((sz - w) / 2) + 1);
rEnd = min(sz, r0 + h - 1);
cEnd = min(sz, c0 + w - 1);
canvas(r0:rEnd, c0:cEnd) = img(1:(rEnd - r0 + 1), 1:(cEnd - c0 + 1));
end

function regions = find_dense_regions(Y, nRegions)
% Greedy grid scan over the normalized t-SNE plane for the most populated
% boxes, skipping boxes that overlap one already picked.
Yn = (Y - min(Y, [], 1)) ./ max(max(Y, [], 1) - min(Y, [], 1), eps);
boxSize = 0.065;
step = 0.04;
candidates = [];
for cx = 0:step:(1 - boxSize)
    for cy = 0:step:(1 - boxSize)
        n = sum(Yn(:, 1) >= cx & Yn(:, 1) < cx + boxSize & Yn(:, 2) >= cy & Yn(:, 2) < cy + boxSize);
        candidates(end + 1, :) = [cx cy n]; %#ok<AGROW>
    end
end
candidates = sortrows(candidates, -3);
regions = {};
picked = zeros(0, 2);
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
