function [knnAcc, purity, nmi] = label_metrics(F, labels, k, nClusters)
%LABEL_METRICS Score how well features separate known classes.
%
%   [KNNACC, PURITY, NMI] = LABEL_METRICS(F, LABELS, K, NCLUSTERS) with F
%   a D x N feature matrix (single, CPU or gpuArray) and LABELS N x 1:
%       KNNACC - leave-one-out K-nearest-neighbour accuracy (cosine
%                similarity, majority vote, ties to the smaller label)
%       PURITY - K-means (NCLUSTERS clusters, on L2-normalized features)
%                purity: the accuracy of giving every cluster its
%                majority label, i.e. the best case for cluster-then-label
%       NMI    - normalized mutual information between clusters and
%                labels (arithmetic-mean normalization), 1 = perfect
%                agreement, ~0 = unrelated

F = F ./ (vecnorm(F, 2, 1) + eps('single'));
labels = cast(labels(:), 'like', F);
N = size(F, 2);

pred = zeros(N, 1, 'like', F);
chunk = 2048; % bounds the chunk x N similarity block in GPU memory
for s = 1:chunk:N
    cols = s:min(N, s + chunk - 1);
    S = F(:, cols)' * F;
    % Cosine similarity is >= -1, so -3 on the diagonal drops each point from its own neighbours.
    S(:, cols) = S(:, cols) - 3 * eye(numel(cols), 'like', S);
    [~, nn] = maxk(S, k, 2);
    pred(cols) = mode(labels(nn), 2);
end
knnAcc = gather(mean(pred == labels));

rng(0);
if isgpuarray(F)
    gpurng(0);
end
cluster = cast(kmeans(F', nClusters, 'Replicates', 5, 'MaxIter', 500), 'like', F);
classes = unique(labels)';
C = single(cluster == 1:nClusters)' * single(labels == classes); % clusters x classes counts
purity = gather(sum(max(C, [], 2)) / N);

P = C / N;
px = sum(P, 2);
py = sum(P, 1);
terms = P .* log(P ./ (px .* py));
terms(P == 0) = 0;
hx = -sum(px(px > 0) .* log(px(px > 0)));
hy = -sum(py(py > 0) .* log(py(py > 0)));
nmi = gather(2 * sum(terms, 'all') / (hx + hy));
end
