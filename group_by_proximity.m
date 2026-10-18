function groupId = group_by_proximity(bbx, marginX, marginY)
%GROUP_BY_PROXIMITY Group bounding boxes [x y w h] whose margin-expanded
%extents overlap in both x and y -- merges a syllable's base stack with a
%floating vowel sign above/below it even when they were assigned to
%different detected line-bands.
%
%   The pairwise overlap test is one n x n comparison (on the GPU when
%   available); groups are the connected components of the overlap
%   graph. GROUPID is a single column of group labels 1..K.
b = single(bbx);
if canUseGPU()
    b = gpuArray(b);
end
x1 = b(:, 1); y1 = b(:, 2); x2 = x1 + b(:, 3); y2 = y1 + b(:, 4);

ox = (x1 - marginX < x2') & (x1' - marginX < x2);
oy = (y1 - marginY < y2') & (y1' - marginY < y2);

groupId = single(conncomp(graph(gather(ox & oy), 'omitselfloops')))';
end
