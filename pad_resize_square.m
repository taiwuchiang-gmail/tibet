function out = pad_resize_square(img, sz)
%PAD_RESIZE_SQUARE Pad a grayscale image to a square canvas (background
%taken from the image's own max value, matching whichever polarity it
%is) and resize to sz x sz, preserving aspect ratio. Returns single, on
%the GPU if IMG is a gpuArray.
img = single(img);
[h, w] = size(img);
s = max(h, w);
canvas = ones(s, s, 'like', img) * max(img(:));
r0 = floor((s - h) / 2) + 1;
c0 = floor((s - w) / 2) + 1;
canvas(r0:r0 + h - 1, c0:c0 + w - 1) = img;
out = imresize(canvas, [sz, sz]);
end
