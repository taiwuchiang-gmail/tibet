% Seeds 2 and 3 of the MNIST grid2 runs with 0/1/10/50 labels per digit
% (seed 1 is in results/sweep_bottleneck_labels.csv), to tell the effect
% of the labels apart from run-to-run noise. Loop state uses sweep* names
% because run_tibet_pipeline is a script sharing this workspace.
sweepRows = {};
for sweepSeed = [2 3]
    for sweepL = [0 1 10 50]
        cfgOverride = struct('dataset', 'digits_mnist', 'bottleneck', 'grid2', ...
            'labelsPerClass', sweepL, 'seed', sweepSeed);
        run_tibet_pipeline;
        close all;
        inv = invariance(end);
        lm = labelMetrics(end);
        sweepRows(end + 1, :) = {char(stamp), cfg.seed, cfg.labelsPerClass, inv.shiftMean, inv.shiftWorst, ...
            lm.knnAcc, lm.kmeansPurity, lm.kmeansNMI, lm.zeroCluster(1), lm.zeroCluster(2), ...
            lm.zeroCluster(3), lm.zeroCluster(4)}; %#ok<SAGROW>
        sweepT = cell2table(sweepRows, 'VariableNames', {'run', 'seed', 'labelsPerClass', 'shiftMean', ...
            'shiftWorst', 'knn', 'purity', 'nmi', 'zc0', 'zc6', 'zc9', 'zcOther'});
        writetable(sweepT, fullfile(projDir, 'results', 'sweep_label_seeds.csv'));
    end
end
disp(sweepT);
