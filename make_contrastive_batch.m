function Xbatch = make_contrastive_batch(X, idx, imgSize)
%MAKE_CONTRASTIVE_BATCH Build a 2N-column minibatch of two independent
%augmented views per sample, ordered [view1(1..N), view2(1..N)] to match
%ntxent_loss's positive-pairing convention (column i pairs with i+N).
%Stays on the GPU when X is a gpuArray.
N = numel(idx);
batch = reshape(X(:, idx), imgSize, imgSize, 1, N);
view1 = reshape(augment_glyph(batch), [], N);
view2 = reshape(augment_glyph(batch), [], N);
Xbatch = [view1, view2];
end
