% Build both digit datasets and save check montages: row = folder label
% (digit 0..9), columns = different fonts / MNIST writers.
projDir = pwd;
printedDir = fullfile(projDir, 'digits', 'printed');
mnistDir = fullfile(projDir, 'digits', 'mnist');
for d = {printedDir, mnistDir}
    if isfolder(d{1}), rmdir(d{1}, 's'); end
end

t0 = tic;
info = make_printed_digits(printedDir);
fprintf('printed: %.0fs\n', toc(t0));
disp(info.rejected);

t0 = tic;
counts = make_mnist_digits(mnistDir, fullfile(projDir, 'digits', 'raw'), "test");
fprintf('mnist: %.0fs, per-digit counts %s\n', toc(t0), mat2str(counts));
expected = [980 1135 1032 1010 982 892 958 1028 974 1009]; % published MNIST test-set counts
fprintf('mnist counts match published test-set counts: %d\n', isequal(counts, expected));

rng(3);
for spec = {{printedDir, 'printed', 24}, {mnistDir, 'mnist', 24}}
    [dirPath, name, nCols] = spec{1}{:};
    tiles = {};
    for d = 0:9
        files = dir(fullfile(dirPath, num2str(d), '*.png'));
        pick = randperm(numel(files), nCols);
        for k = pick
            tiles{end + 1} = imread(fullfile(files(k).folder, files(k).name)); %#ok<SAGROW>
        end
    end
    figure(1); clf;
    montage(tiles, 'Size', [10 nCols], 'BorderSize', [1 1], 'BackgroundColor', [0.8 0.2 0.2]);
    title(sprintf('%s digits: row k = folder "k" (random %d per row)', name, nCols));
    exportgraphics(figure(1), fullfile(projDir, 'scratch', sprintf('digits_%s_check.png', name)), 'Resolution', 120);
end

% Printed set, every accepted font for one digit, to spot oddball fonts.
files = dir(fullfile(printedDir, '5', '*.png'));
figure(2); clf;
montage(fullfile({files.folder}, {files.name}), 'BorderSize', [1 1], 'BackgroundColor', [0.8 0.2 0.2]);
title(sprintf('printed "5" in all %d accepted fonts', numel(files)));
exportgraphics(figure(2), fullfile(projDir, 'scratch', 'digits_printed_all_fonts_5.png'), 'Resolution', 120);
