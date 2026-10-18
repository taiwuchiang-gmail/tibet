% Verify the batched GPU augmentation and the rewritten NT-Xent loss
% against the original CPU implementations (scratch/ref_old).
addpath(fullfile(pwd, 'scratch', 'ref_old'));
rng(1); gpurng(1);

%% 1. NT-Xent: same value and gradient as the original
Zc = randn(16, 128, 'single');
Zc = Zc ./ vecnorm(Zc);
lossOld = ntxent_loss_old(dlarray(double(Zc), 'CB'), 0.1);
lossNew = ntxent_loss(dlarray(gpuArray(Zc), 'CB'), 0.1);
fprintf('NT-Xent old(double CPU)=%.6f  new(single GPU)=%.6f  type=%s gpu=%d\n', ...
    extractdata(lossOld), gather(extractdata(lossNew)), underlyingType(lossNew), isgpuarray(extractdata(lossNew)));
[~, gOld] = dlfeval(@(z) deal(ntxent_loss_old(z, 0.1), dlgradient(ntxent_loss_old(z, 0.1), z)), dlarray(double(Zc), 'CB'));
[~, gNew] = dlfeval(@(z) deal(ntxent_loss(z, 0.1), dlgradient(ntxent_loss(z, 0.1), z)), dlarray(gpuArray(Zc), 'CB'));
fprintf('NT-Xent max |grad diff| = %.2e\n', max(abs(double(gather(extractdata(gNew))) - extractdata(gOld)), [], 'all'));

%% 2. Augmentation: type, range, and per-sample change vs. the original
[Xp, ~] = load_crop_dataset({fullfile(pwd, 'printed_transcription_crops')}, 32, Inf);
imgs = reshape(Xp, 32, 32, 1, []);
imgsG = gpuArray(imgs);
A = augment_glyph(imgsG);
fprintf('augment out: %s/%s, size %s, range [%.3f %.3f]\n', class(A), underlyingType(A), ...
    mat2str(size(A)), gather(min(A, [], 'all')), gather(max(A, [], 'all')));

nCmp = 400;
dOld = zeros(nCmp, 1, 'single');
inkOld = zeros(nCmp, 1, 'single');
for i = 1:nCmp
    a = single(augment_glyph_old(double(imgs(:, :, 1, i))));
    dOld(i) = mean(abs(a - imgs(:, :, 1, i)), 'all');
    inkOld(i) = sum(1 - a, 'all') / sum(1 - imgs(:, :, 1, i), 'all');
end
An = gather(A(:, :, :, 1:nCmp));
dNew = squeeze(mean(abs(An - imgs(:, :, :, 1:nCmp)), [1 2]));
inkNew = squeeze(sum(1 - An, [1 2]) ./ sum(1 - imgs(:, :, :, 1:nCmp), [1 2]));
fprintf('mean |aug - orig| per pixel:  old %.4f (sd %.4f)   new %.4f (sd %.4f)\n', mean(dOld), std(dOld), mean(dNew), std(dNew));
fprintf('ink mass ratio aug/orig:      old %.3f (sd %.3f)   new %.3f (sd %.3f)\n', mean(inkOld), std(inkOld), mean(inkNew), std(inkNew));

%% 3. Visual check: original | 4 old augs | 4 new augs
sel = [3 50 120 300 520 800];
tiles = {};
for s = sel
    tiles{end + 1} = imgs(:, :, 1, s); %#ok<SAGROW>
    for k = 1:4
        tiles{end + 1} = single(augment_glyph_old(double(imgs(:, :, 1, s)))); %#ok<SAGROW>
    end
    newA = gather(augment_glyph(repmat(imgsG(:, :, :, s), 1, 1, 1, 4)));
    for k = 1:4
        tiles{end + 1} = newA(:, :, 1, k); %#ok<SAGROW>
    end
end
figure(1);
montage(cellfun(@(t) imresize(t, 3, 'nearest'), tiles, 'UniformOutput', false), ...
    'Size', [numel(sel) 9], 'BorderSize', [2 2], 'BackgroundColor', 'red');
title('col 1: original | cols 2-5: old CPU augment | cols 6-9: new batched GPU augment');
exportgraphics(figure(1), fullfile(pwd, 'scratch', 'augmentation_check_gpu.png'), 'Resolution', 150);

%% 4. Speed for one contrastive batch (64 samples -> 128 views)
idx = 1:64;
tOld = timeit(@() local_old_batch(imgs, idx));
tNew = gputimeit(@() make_contrastive_batch_img(imgsG, idx));
fprintf('contrastive batch build: old CPU loop %.1f ms, new GPU batched %.2f ms (%.0fx)\n', tOld * 1e3, tNew * 1e3, tOld / tNew);

function Xb = local_old_batch(imgs, idx)
N = numel(idx);
v1 = zeros(32, 32, 1, N, 'single');
v2 = v1;
for i = 1:N
    img = double(imgs(:, :, 1, idx(i)));
    v1(:, :, 1, i) = augment_glyph_old(img);
    v2(:, :, 1, i) = augment_glyph_old(img);
end
Xb = cat(4, v1, v2);
end
