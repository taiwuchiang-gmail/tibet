% Full runs: gap vs grid2 bottleneck on all three datasets, then
% semi-supervised SupCon with 1/10/50 labels per digit on MNIST.
% run_tibet_pipeline is a script sharing this workspace, so loop state
% uses sweep* names it never touches.
sweepCfgs = {
    struct('dataset', 'digits_printed')
    struct('dataset', 'digits_printed', 'bottleneck', 'grid2')
    struct('dataset', 'digits_mnist')
    struct('dataset', 'digits_mnist', 'bottleneck', 'grid2')
    struct('resegment', false)
    struct('resegment', false, 'bottleneck', 'grid2')
    struct('dataset', 'digits_mnist', 'bottleneck', 'gap', 'labelsPerClass', 10)
    struct('dataset', 'digits_mnist', 'bottleneck', 'grid2', 'labelsPerClass', 1)
    struct('dataset', 'digits_mnist', 'bottleneck', 'grid2', 'labelsPerClass', 10)
    struct('dataset', 'digits_mnist', 'bottleneck', 'grid2', 'labelsPerClass', 50)
    struct('dataset', 'digits_printed', 'bottleneck', 'grid2', 'labelsPerClass', 10)
    };
sweepRows = {};
for sweepK = 1:numel(sweepCfgs)
    cfgOverride = sweepCfgs{sweepK};
    run_tibet_pipeline;
    close all;
    inv = invariance(end);
    if isempty(labelMetrics)
        lm = struct('knnAcc', NaN, 'kmeansPurity', NaN, 'kmeansNMI', NaN, 'zeroCluster', NaN(1, 4));
    else
        lm = labelMetrics(end);
    end
    sweepRows(end + 1, :) = {char(stamp), cfg.dataset, cfg.bottleneck, cfg.labelsPerClass, ...
        inv.shiftMean, inv.shiftWorst, inv.scaleMean, inv.scaleWorst, lm.knnAcc, lm.kmeansPurity, lm.kmeansNMI, ...
        lm.zeroCluster(1), lm.zeroCluster(2), lm.zeroCluster(3), lm.zeroCluster(4)}; %#ok<SAGROW>
    sweepT = cell2table(sweepRows, 'VariableNames', {'run', 'dataset', 'bottleneck', 'labelsPerClass', ...
        'shiftMean', 'shiftWorst', 'scaleMean', 'scaleWorst', 'knn', 'purity', 'nmi', 'zc0', 'zc6', 'zc9', 'zcOther'});
    writetable(sweepT, fullfile(projDir, 'results', 'sweep_bottleneck_labels.csv'));
end
disp(sweepT);
