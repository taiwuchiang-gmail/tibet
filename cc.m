I=imread('1-1-2b.jpg');
T = graythresh(I);
BW = imbinarize(rgb2gray(I),"global");
BW1=~BW;
BW_edge = edge(BW1,'canny');
figure(5); imshow(BW_edge);
[H,theta,rho] = hough(BW_edge,'RhoResolution',0.3,'Theta',-90:0.3:89);
P = houghpeaks(H,100,'threshold',ceil(0.1*max(H(:))));

lines = houghlines(BW_edge,theta,rho,P,'FillGap',3,'MinLength',30);
tmp = size(lines);
num_of_lines = tmp(2);
%     
figure(7), imshow(BW1, []); hold on
max_len = 0;
for k = 1:num_of_lines
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',1,'Color','green');
    
    %Plot beginnings and ends of lines
    %plot(xy(1,1),xy(1,2),'x','LineWidth',1,'Color','yellow');
    %plot(xy(2,1),xy(2,2),'x','LineWidth',1,'Color','red');
    
    %Determine the endpoints of the longest line segment
    len = norm(lines(k).point1 - lines(k).point2);
    if ( len > max_len)
        max_len = len;
        xy_long = xy;
    end

end

%highlight the longest line segment
%plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','red');

%find coordinates of all intersections
[height, width] = size(BW1);
C = nchoosek(1:num_of_lines,2);
num_of_permu = size(C);
X=[];
Y=[];
for i=1 : num_of_permu(1)
    [x, y] = line_intersect_coord(lines(C(i,1)), lines(C(i,2)));
    if (isfinite(x) && isfinite(y))
        X = [X x];
        Y = [Y y];
    end
end
    
plot(X,Y,'x','LineWidth',2,'Color','red');
figure(7), hold off
cut = 0;
for k =1:length(X)
    if (X(k) <= 69 && Y(k) <= 32) && (X(k) > 32 && Y(k) > 3)
        L_T_X = X(k);
        L_T_Y = Y(k);
    elseif (X(k) <= 1236 && Y(k) <= 28) && (X(k) > 1210 && Y(k) > 3)
        R_T_X= X(k);
        R_T_Y= Y(k);
    elseif (X(k) <= 1236 && Y(k) <= 305) && (X(k) > 1210 && Y(k) > 281)
        R_B_X= X(k);
        R_B_Y= Y(k);
    elseif (X(k) <= 60 && Y(k) <= 304) && (X(k) > 38 && Y(k) > 289)
        L_B_X= X(k);
        L_B_Y= Y(k);
    end
end

cut_x = 9; %pixels to cut the black boarder
cut_y = 3; %pixels to cut the black boarder
bx = [L_T_X+cut_x; L_T_Y+cut_y; R_T_X - L_T_X - 2*cut_x; L_B_Y - L_T_Y - 2*cut_y];
BW_crop=imcrop(BW1,bx);
figure(10); imshow(BW_crop, []);

    

CC = bwconncomp(BW_crop,4);

BW2=zeros(size(BW_crop));

s = regionprops(CC,"Area","Extent","BoundingBox");
areas = cat(1,s.Area);
extent = cat(1,s.Extent);
bbx = cat(1,s.BoundingBox);
ratio = bbx(:,3)./bbx(:,4);
idx = find(areas>30 & areas<500 & ratio>.1 & ratio<10 & extent>0.1 & bbx(:,3)<100 & bbx(:,4)<100 & (bbx(:,3).*bbx(:,4))>50);

BBX = [];
len = length(idx);
for k = 1: len
    BW2(CC.PixelIdxList{idx(k)}) = 1;
    BBX = [BBX; bbx(idx,:)];
end
figure(2); imshow(insertShape(BW2,"rectangle",BBX,LineWidth=1, Color=[0 255 0]), []);



