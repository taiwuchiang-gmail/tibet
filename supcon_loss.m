function loss = supcon_loss(Z, tau, labels)
%SUPCON_LOSS Supervised contrastive loss (Khosla et al. 2020, "L_out")
%for a batch in which only some samples have known labels.
%
%   Z is D x 2N: L2-normalized embeddings, columns i and i+N are the two
%   augmented views of sample i (same pairing as ntxent_loss). LABELS is
%   1 x N: the class of each sample, or -1 if unknown.
%
%   The positives of an anchor are its other view plus every view of a
%   sample with the same known label; every other column is a negative
%   in the softmax denominator, so known different-class samples are
%   pushed apart. The loss averages -log p(positive) over each anchor's
%   positives. With no known labels this is exactly ntxent_loss.

Z = stripdims(Z);
N2 = size(Z, 2);
N = N2 / 2;
proto = extractdata(Z(1)); % eye/cast 'like' a dlarray is unsupported

sim = (Z' * Z) / tau;
self = eye(N2, 'like', proto);
logits = sim - self * 1e9;
m = max(logits, [], 2);
logProb = logits - (m + log(sum(exp(logits - m), 2)));

lab = cast([labels(:)', labels(:)'], 'like', proto);
partner = circshift(self, N, 2) > 0; % column i+N for row i, and i for row i+N
sameKnown = (lab' == lab) & (lab' >= 0) & ~self;
pos = cast(partner | sameKnown, 'like', proto);

loss = mean(-sum(pos .* logProb, 2) ./ sum(pos, 2));
end
