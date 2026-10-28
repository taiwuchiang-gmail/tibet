% Check label_metrics against cases with known answers and against
% independent CPU double-precision references (fitcknn-free brute force
% kNN, and a plain-loop purity/NMI on the same K-means assignment).
rng(5);
N = 3000;
labels = single(randi([0 9], N, 1));

% 1. Perfectly separable: one-hot class + tiny noise -> all scores ~1.
F = single(labels' == (0:9)') + 0.01 * randn(10, N, 'single');
[a, p, n] = label_metrics(gpuArray(F), labels, 5, 10);
fprintf('separable:  kNN %.3f purity %.3f NMI %.3f   (expect 1, 1, 1)\n', a, p, n);

% 2. Pure noise -> kNN ~0.1, NMI ~0.
F = randn(16, N, 'single');
[a, p, n] = label_metrics(gpuArray(F), labels, 5, 10);
fprintf('noise:      kNN %.3f purity %.3f NMI %.3f   (expect ~0.1, low, ~0)\n', a, p, n);

% 3. Partially separable, compared with an independent CPU double reference.
F = 0.6 * single(labels' == (0:9)') + randn(10, N, 'single') * 0.35;
[a, p, n] = label_metrics(gpuArray(F), labels, 5, 10);
Fd = double(F) ./ vecnorm(double(F));
S = Fd' * Fd;
S(1:N + 1:end) = -Inf;
[~, ord] = sort(S, 2, 'descend');
predRef = mode(double(labels(ord(:, 1:5))), 2);
aRef = mean(predRef == double(labels));
rng(0); gpurng(0);
cl = gather(kmeans(gpuArray(F ./ (vecnorm(F) + eps('single')))', 10, 'Replicates', 5, 'MaxIter', 500));
pRef = 0; mi = 0;
for c = unique(cl)'
    pRef = pRef + max(histcounts(labels(cl == c), -0.5:1:9.5));
end
pRef = pRef / N;
for c = unique(cl)'
    for y = 0:9
        pxy = mean(cl == c & labels == y);
        if pxy > 0
            mi = mi + pxy * log(pxy / (mean(cl == c) * mean(labels == y)));
        end
    end
end
hc = -sum(arrayfun(@(c) mean(cl == c) * log(mean(cl == c)), unique(cl)'));
hy = -sum(arrayfun(@(y) mean(labels == y) * log(mean(labels == y)), 0:9));
nRef = 2 * mi / (hc + hy);
fprintf('partial:    kNN %.4f vs ref %.4f | purity %.4f vs ref %.4f | NMI %.4f vs ref %.4f\n', a, aRef, p, pRef, n, nRef);
