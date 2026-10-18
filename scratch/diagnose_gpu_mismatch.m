% Which GPU-capable step in segment_page differs from its CPU counterpart?
scanList = dir(fullfile(pwd, 'samples_adarshah', '*.jpg'));
pages = [{'1-1-2b.jpg'}, fullfile({scanList.folder}, {scanList.name}), {'printed_transcription_test.png'}];
for i = 1:numel(pages)
    I = imread(pages{i});
    if size(I, 3) == 3, gray = rgb2gray(I); else, gray = I; end
    BW = imbinarize(gray, 'global');
    if nnz(BW) <= numel(BW) / 2, ink = BW; else, ink = ~BW; end
    [H, theta] = hough(edge(ink, 'canny'), 'RhoResolution', 0.5, 'Theta', -45:0.2:44.8);
    P = houghpeaks(H, 40, 'threshold', ceil(0.2 * max(H(:))));
    ang = single(median(theta(P(:, 2))));
    rotDiff = NaN; grayDiff = NaN;
    if abs(ang) > 0.05
        rc = imrotate(ink, ang, 'nearest', 'loose');
        rg = gather(imrotate(gpuArray(ink), ang, 'nearest', 'loose'));
        rotDiff = nnz(rc ~= rg);
        gc = imrotate(gray, ang, 'bilinear', 'loose');
        gg = gather(imrotate(gpuArray(gray), ang, 'bilinear', 'loose'));
        grayDiff = nnz(gc ~= gg);
        ink = rc;
    end
    se = strel('line', round(size(ink, 1) * 0.5), 90);
    openDiff = nnz(imopen(ink, se) ~= gather(imopen(gpuArray(ink), se)));
    se2 = strel('line', round(size(ink, 2) * 0.3), 0);
    openDiff2 = nnz(imopen(ink, se2) ~= gather(imopen(gpuArray(ink), se2)));
    distDiff = max(abs(bwdist(~ink) - gather(bwdist(~gpuArray(ink)))), [], 'all');
    fprintf('%-32s angle %6.2f | rotate ink diff %g px, gray diff %g px | imopen vert %d, horiz %d px | bwdist max diff %g\n', ...
        pages{i}(max(1, end-30):end), ang, rotDiff, grayDiff, openDiff, openDiff2, distDiff);
end
