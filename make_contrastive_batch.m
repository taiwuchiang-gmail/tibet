function Xbatch = make_contrastive_batch(X, idx, imgSize)
%MAKE_CONTRASTIVE_BATCH Build a 2N-column minibatch of two independent
%augmented views per sample, ordered [view1(1..N), view2(1..N)] to match
%ntxent_loss's positive-pairing convention (column i pairs with i+N).
N = numel(idx);
D = size(X, 1);
view1 = zeros(D, N, 'single');
view2 = zeros(D, N, 'single');
for i = 1:N
    img = reshape(X(:, idx(i)), imgSize, imgSize);
    a1 = augment_glyph(img);
    a2 = augment_glyph(img);
    view1(:, i) = single(a1(:));
    view2(:, i) = single(a2(:));
end
Xbatch = [view1, view2];
end
