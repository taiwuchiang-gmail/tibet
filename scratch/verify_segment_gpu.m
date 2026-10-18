% Re-segment every page with the single/GPU segment_page and compare the
% crops against the existing folders, which were produced by the
% original double-precision CPU version.
projDir = pwd;
scanList = dir(fullfile(projDir, 'samples_adarshah', '*.jpg'));
pages = [{fullfile(projDir, '1-1-2b.jpg')}, fullfile({scanList.folder}, {scanList.name}), ...
    {fullfile(projDir, 'printed_transcription_test.png')}];
refDirs = [{fullfile(projDir, '1-1-2b_crops')}, ...
    cellfun(@(n) fullfile(projDir, 'samples_adarshah_crops', erase(n, '.jpg')), {scanList.name}, 'UniformOutput', false), ...
    {fullfile(projDir, 'printed_transcription_crops')}];
tmpRoot = fullfile(tempdir, 'tibet_seg_verify');

for useGPU = [false true]
    if isfolder(tmpRoot)
        rmdir(tmpRoot, 's');
    end
    segment_page(pages{1}, 'OutputDir', fullfile(tmpRoot, 'warmup'), 'UseGPU', useGPU);
    rmdir(fullfile(tmpRoot, 'warmup'), 's');
    t0 = tic;
    nSame = 0; nPixDiff = 0; maxDiff = 0; nMissing = 0; nExtra = 0; nTotal = 0;
    for i = 1:numel(pages)
        out = fullfile(tmpRoot, sprintf('p%02d', i));
        r = segment_page(pages{i}, 'OutputDir', out, 'UseGPU', useGPU);
        assert(isa(r(1).BoundingBox, 'single') && isa(r(1).Line, 'single'), 'outputs not single');
        refNames = {dir(fullfile(refDirs{i}, '*.png')).name};
        newNames = {dir(fullfile(out, '*.png')).name};
        nMissing = nMissing + numel(setdiff(refNames, newNames));
        nExtra = nExtra + numel(setdiff(newNames, refNames));
        common = intersect(refNames, newNames);
        nTotal = nTotal + numel(refNames);
        for k = 1:numel(common)
            a = imread(fullfile(refDirs{i}, common{k}));
            b = imread(fullfile(out, common{k}));
            if isequal(size(a), size(b)) && isequal(a, b)
                nSame = nSame + 1;
            else
                nPixDiff = nPixDiff + 1;
                if isequal(size(a), size(b))
                    maxDiff = max(maxDiff, max(abs(int16(a(:)) - int16(b(:)))));
                else
                    maxDiff = Inf;
                end
            end
        end
    end
    fprintf('UseGPU=%d: %.2fs for %d pages | ref crops %d | identical %d | pixel-different %d (max |diff| %g) | missing %d | extra %d\n', ...
        useGPU, toc(t0), numel(pages), nTotal, nSame, nPixDiff, maxDiff, nMissing, nExtra);
end
rmdir(tmpRoot, 's');
