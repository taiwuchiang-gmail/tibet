function groupId = group_by_proximity(bbx, marginX, marginY)
%GROUP_BY_PROXIMITY Union-find grouping of bounding boxes [x y w h] whose
%margin-expanded extents overlap in both x and y -- merges a syllable's
%base stack with a floating vowel sign above/below it even when they
%were assigned to different detected line-bands.
n = size(bbx, 1);
x1 = bbx(:,1); y1 = bbx(:,2); x2 = bbx(:,1)+bbx(:,3); y2 = bbx(:,2)+bbx(:,4);
parent = (1:n)';

for i = 1:n
    for j = (i+1):n
        ox = (x1(i)-marginX < x2(j)) && (x1(j)-marginX < x2(i));
        oy = (y1(i)-marginY < y2(j)) && (y1(j)-marginY < y2(i));
        if ox && oy
            ri = find_root(parent, i);
            rj = find_root(parent, j);
            if ri ~= rj
                parent(ri) = rj;
            end
        end
    end
end

groupId = zeros(n,1);
for i = 1:n
    groupId(i) = find_root(parent, i);
end
end

function r = find_root(parent, i)
r = i;
while parent(r) ~= r
    r = parent(r);
end
end
