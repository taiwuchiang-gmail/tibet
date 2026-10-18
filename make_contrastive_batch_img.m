function Xbatch = make_contrastive_batch_img(imgs, idx)
%MAKE_CONTRASTIVE_BATCH_IMG Like make_contrastive_batch.m but for a
%convolutional encoder: returns a HxWx1x2N image array (instead of a
%(H*W)x2N flattened-vector matrix), two independent augmented views per
%sample, ordered [view1(1..N), view2(1..N)] to match ntxent_loss's
%positive-pairing convention (column i pairs with i+N). Stays on the GPU
%when IMGS is a gpuArray.
batch = imgs(:, :, :, idx);
Xbatch = cat(4, augment_glyph(batch), augment_glyph(batch));
end
