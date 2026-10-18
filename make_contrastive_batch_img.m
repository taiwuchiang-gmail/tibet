function Xbatch = make_contrastive_batch_img(imgs, idx)
%MAKE_CONTRASTIVE_BATCH_IMG Like make_contrastive_batch.m but for a
%convolutional encoder: returns a 32x32x1x2N image array (instead of a
%1024x2N flattened-vector matrix), two independent augmented views per
%sample, ordered [view1(1..N), view2(1..N)] to match ntxent_loss's
%positive-pairing convention (column i pairs with i+N).
N = numel(idx);
[h, w, c, ~] = size(imgs);
view1 = zeros(h, w, c, N, 'single');
view2 = zeros(h, w, c, N, 'single');
for i = 1:N
    img = imgs(:, :, 1, idx(i));
    a1 = augment_glyph(img);
    a2 = augment_glyph(img);
    view1(:, :, 1, i) = single(a1);
    view2(:, :, 1, i) = single(a2);
end
Xbatch = cat(4, view1, view2);
end
