% Why do 6s and 9s land in the 0 cluster? Test whether netCC's global
% average pooling (which keeps *what* stroke patterns occur but not
% *where*) is the cause, by reading the same trained network's features
% before pooling at coarser/finer spatial grids. No retraining.
runDir = fullfile(pwd, 'results', '20261023_043713_digits_mnist');
S = load(fullfile(runDir, 'models.mat'), 'netCC');
netCC = S.netCC;

[X, meta] = load_crop_dataset({fullfile(pwd, 'digits', 'mnist')}, 32, Inf);
[~, names] = cellfun(@(p) fileparts(fileparts(p)), {meta.Path}, 'UniformOutput', false);
labels = single(str2double(names))';
imgs = gpuArray(reshape(X, 32, 32, 1, []));
N = numel(labels);

A = predict(netCC, imgs, 'Outputs', 'relu3'); % 8 x 8 x 32 x N, input to the global average pool
pool2 = squeeze(mean(reshape(A, 4, 2, 4, 2, 32, N), [1 3])); % 2 x 2 x 32 x N
feats = {
    'netCC global avg pool (32-d, current)', squeeze(predict(netCC, imgs, 'Outputs', 'bottleneck'))
    'netCC 2x2 grid pool (128-d)',           reshape(pool2, [], N)
    'netCC 8x8 map, no pooling (2048-d)',    reshape(A, [], N)
    'raw pixels (1024-d)',                   reshape(1 - imgs, [], N)
    };

fprintf('%-40s %7s %9s %9s %9s %8s %6s\n', 'features', 'kNN', '6 -> 0', '9 -> 0', '0 -> 6/9', 'purity', 'NMI');
for m = 1:size(feats, 1)
    F = feats{m, 2};
    pred = knn_predict(F, labels, 5);
    six = labels == 6; nine = labels == 9; zero = labels == 0;
    [~, pur, nmi] = label_metrics(F, labels, 5, 10);
    fprintf('%-40s %7.3f %8.1f%% %8.1f%% %8.1f%% %8.3f %6.3f\n', feats{m, 1}, mean(pred == labels), ...
        100 * mean(pred(six) == 0), 100 * mean(pred(nine) == 0), 100 * mean(pred(zero) == 6 | pred(zero) == 9), pur, nmi);
end

% Composition of the K-means cluster(s) where most 0s end up.
fprintf('\nK-means k=10: the cluster holding the most 0s, and what else is in it\n');
for m = 1:size(feats, 1)
    F = feats{m, 2};
    F = F ./ (vecnorm(F, 2, 1) + eps('single'));
    rng(0); gpurng(0);
    cl = gather(kmeans(F', 10, 'Replicates', 5, 'MaxIter', 500));
    zc = mode(cl(labels == 0));
    in = cl == zc;
    fprintf('%-40s size %4d | 0: %4d (%.0f%% of all 0s) | 6: %3d | 9: %3d | other: %3d\n', feats{m, 1}, nnz(in), ...
        nnz(in & labels == 0), 100 * mean(cl(labels == 0) == zc), nnz(in & labels == 6), nnz(in & labels == 9), ...
        nnz(in & ~ismember(labels, [0 6 9])));
end

function pred = knn_predict(F, labels, k)
F = F ./ (vecnorm(F, 2, 1) + eps('single'));
labelsG = cast(labels(:), 'like', F);
N = size(F, 2);
pred = zeros(N, 1, 'like', F);
for s = 1:2048:N
    cols = s:min(N, s + 2047);
    Sim = F(:, cols)' * F;
    Sim(:, cols) = Sim(:, cols) - 3 * eye(numel(cols), 'like', Sim);
    [~, nn] = maxk(Sim, k, 2);
    pred(cols) = mode(labelsG(nn), 2);
end
pred = gather(pred);
end
