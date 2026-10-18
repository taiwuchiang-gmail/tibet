function [x, y] = line_intersect_coord(line1, line2)

    p11.x = line1.point1(1);
    p11.y = line1.point1(2);

    p12.x = line1.point2(1);
    p12.y = line1.point2(2);

    p21.x = line2.point1(1);
    p21.y = line2.point1(2);

    p22.x = line2.point2(1);
    p22.y = line2.point2(2);

	a1 = p11.y - p12.y;
	b1 = p12.x - p11.x;
	c1 = p11.x * p12.y - p12.x * p11.y;

	a2 = p21.y - p22.y;
	b2 = p22.x - p21.x;
	c2 = p21.x * p22.y - p22.x * p21.y;

	temp = (a1*b2-a2*b1);
    
 	x = round((b1*c2 - b2*c1)/temp);
	y = round((c1*a2 - c2*a1)/temp);

end