% supcon_loss: (1) equals ntxent_loss when nothing is labeled, (2) matches
% a plain-loop double-precision reference with labels, (3) gradients agree.
rng(2);
N = 24;
Z = randn(8, 2 * N, 'single');
Z = Z ./ vecnorm(Z);
tau = 0.1;
noLabels = -ones(1, N, 'single');
labels = single(randi([0 3], 1, N));
labels(rand(1, N) < 0.5) = -1; % about half unknown

zG = dlarray(gpuArray(Z), 'CB');
fprintf('no labels: supcon %.6f  ntxent %.6f\n', gather(extractdata(supcon_loss(zG, tau, noLabels))), ...
    gather(extractdata(ntxent_loss(zG, tau))));

% Reference: loop over anchors, double precision.
Zd = double(Z);
lab = double([labels, labels]);
ref = 0;
for i = 1:2 * N
    others = setdiff(1:2 * N, i);
    denom = sum(exp(Zd(:, i)' * Zd(:, others) / tau));
    partner = mod(i - 1 + N, 2 * N) + 1;
    P = partner;
    if lab(i) >= 0
        P = union(P, others(lab(others) == lab(i)));
    end
    ref = ref + mean(-log(exp(Zd(:, i)' * Zd(:, P) / tau) / denom));
end
ref = ref / (2 * N);
fprintf('with labels: supcon %.6f  reference %.6f\n', gather(extractdata(supcon_loss(zG, tau, labels))), ref);

[~, g] = dlfeval(@(z) deal(supcon_loss(z, tau, labels), dlgradient(supcon_loss(z, tau, labels), z)), zG);
fd = zeros(5, 1);
an = zeros(5, 1);
for k = 1:5
    e = zeros(size(Zd));
    e(k) = 1e-6;
    fd(k) = (sc_ref(Zd + e, labels, tau) - sc_ref(Zd - e, labels, tau)) / 2e-6;
    an(k) = gather(extractdata(g(k)));
end
fprintf('gradient: max |analytic - finite diff| = %.2e (scale %.2e)\n', max(abs(an - fd)), max(abs(fd)));

function L = sc_ref(Zd, labels, tau)
N = size(Zd, 2) / 2;
lab = double([labels, labels]);
L = 0;
for i = 1:2 * N
    others = setdiff(1:2 * N, i);
    denom = sum(exp(Zd(:, i)' * Zd(:, others) / tau));
    P = mod(i - 1 + N, 2 * N) + 1;
    if lab(i) >= 0
        P = union(P, others(lab(others) == lab(i)));
    end
    L = L + mean(-log(exp(Zd(:, i)' * Zd(:, P) / tau) / denom));
end
L = L / (2 * N);
end
