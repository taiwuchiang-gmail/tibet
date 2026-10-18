function loss = ntxent_loss(Z, tau)
%NTXENT_LOSS Normalized temperature-scaled cross-entropy (SimCLR-style
%contrastive) loss.
%
%   Z is D x 2N: L2-normalized embeddings, where columns 1..N and
%   (N+1)..2N are the two augmented views of the same N underlying
%   samples (view pairing is index i <-> i+N). For each anchor column,
%   the loss maximizes similarity to its one positive (the other view of
%   the same sample) relative to all 2N-2 other columns in the batch
%   (negatives) -- that relative-to-everything-else term is what
%   actively repels different samples' embeddings apart, not just what
%   pulls matching pairs together.

Z = stripdims(Z); % plain matrix algebra below; dlarray dim labels (e.g. 'CB') block transpose/matmul between differently-labeled dims

N2 = size(Z, 2);
N = N2 / 2;

sim = (Z' * Z) / tau; % 2N x 2N cosine-similarity logits (Z columns are unit-norm)

selfMask = eye(N2) * 1e9;
logits = sim - selfMask; % exclude self-similarity from the softmax denominator

posIdx = [(N + 1):N2, 1:N]; % column index of each row's positive partner

m = max(logits, [], 2);
logsumexp = m + log(sum(exp(logits - m), 2));

linIdx = sub2ind([N2, N2], (1:N2)', posIdx(:));
posLogits = logits(linIdx);

loss = mean(logsumexp(:) - posLogits);
end
