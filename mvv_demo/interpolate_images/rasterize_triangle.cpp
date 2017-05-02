#include "rasterize_triangle.h"
#include <iostream>

using namespace std; 
using namespace cv;

/*
vector<pair<int, int>> adrians(int x0, int y0, int x1, int y1) {
	if (x0 == x1) {
		if (y0 == y1) {
			//return trivial
		}
		else {
			//return straight line
		}
	}
	else {
		if (y0 == y1) {
			//return straight line
		}
		else {
			//return tilted line
			int dx = x1 - x0;
			int dy = y1 - y0;
		}
	}
}
*/

vector<pair<int, int>> bresenham(int x0, int y0, int x1, int y1) {
	vector<pair<int, int>> coordinates;
	
	int new_x, new_y;
	int dx, dy, dx_abs, dy_abs;
	int p_x, p_y;
	int err_x, err_y;
	dx = x1 - x0;
	dy = y1 - y0;
	dx_abs = fabs(dx);
	dy_abs = fabs(dy);
	p_x = (dy_abs - dx_abs) << 1;
	p_y = (dx_abs - dy_abs) << 1;

	if (dy_abs <= dx_abs) {
		if (dx >= 0) {
			new_x = x0;
			new_y = y0;
			err_x = x1;
		}
		else {
			new_x = x1;
			new_y = y1;
			err_x = x0;
		}
		coordinates.push_back(make_pair(new_x, new_y));
		cout << "pixel at (" << new_x << "," << new_y << ")" << endl;
		for (int i = 0; new_x < err_x; ++i) {
			new_x++;
			if (p_x < 0) {
				p_x = p_x + (dy_abs << 1);
			}
			else {
				if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
					new_y++;
				}
				else {
					new_y--;
				}
				p_x = p_x + ((dy_abs - dx_abs) << 1);
			}
			coordinates.push_back(make_pair(new_x, new_y));
			cout << "pixel at (" << new_x << "," << new_y << ")" << endl;
		}//end for
	}//end if
	else {
		if (dy >= 0) {
			new_x = x0;
			new_y = y0;
			err_y = y1;
		}
		else {
			new_x = x1;
			new_y = y1;
			err_y = y0;
		}
		coordinates.push_back(make_pair(new_x, new_y));
		cout << "pixel at (" << new_x << "," << new_y << ")" << endl;
		for (int i = 0; new_y < err_y; ++i) {
			new_y++;
			if (p_y <= 0) {
				p_y = p_y + (dx_abs << 1);
			}
			else {
				if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
					new_x++;
				}
				else {
					new_x--;
				}
				p_y = p_y + ((dx_abs - dy_abs) << 1);
			}
			coordinates.push_back(make_pair(new_x, new_y));
			cout << "pixel at (" << new_x << "," << new_y << ")" << endl;
		}//end for
	}//end else
	return coordinates;
}

vector<pair<int, int>> bresenham2(int x1, int y1, int x2, int y2) {
	vector<pair<int, int>> coordinates;
	int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;
	dx = x2 - x1;
	dy = y2 - y1;
	dx1 = fabs(dx);
	dy1 = fabs(dy);
	px = 2 * dy1 - dx1;
	py = 2 * dx1 - dy1;
	if (dy1 <= dx1)
	{
		if (dx >= 0)
		{
			x = x1;
			y = y1;
			xe = x2;
		}
		else
		{
			x = x2;
			y = y2;
			xe = x1;
		}
		coordinates.push_back(make_pair(x, y));
		//cout << "pixel at (" << x << "," << y << ")" << endl;
		for (i = 0; x<xe; i++)
		{
			x = x + 1;
			if (px<0)
			{
				px = px + 2 * dy1;
			}
			else
			{
				if ((dx<0 && dy<0) || (dx>0 && dy>0))
				{
					y = y + 1;
				}
				else
				{
					y = y - 1;
				}
				px = px + 2 * (dy1 - dx1);
			}
			coordinates.push_back(make_pair(x, y));
			//cout << "pixel at (" << x << "," << y << ")" << endl;
		}
	}
	else
	{
		if (dy >= 0)
		{
			x = x1;
			y = y1;
			ye = y2;
		}
		else
		{
			x = x2;
			y = y2;
			ye = y1;
		}
		coordinates.push_back(make_pair(x, y));
		//cout << "pixel at (" << x << "," << y << ")" << endl;
		for (i = 0; y<ye; i++)
		{
			y = y + 1;
			if (py <= 0)
			{
				py = py + 2 * dx1;
			}
			else
			{
				if ((dx<0 && dy<0) || (dx>0 && dy>0))
				{
					x = x + 1;
				}
				else
				{
					x = x - 1;
				}
				py = py + 2 * (dx1 - dy1);
			}
			coordinates.push_back(make_pair(x, y));
			//cout << "pixel at (" << x << "," << y << ")" << endl;
		}
	}
	return coordinates;
}

void triangle_pixels(Point2f a, Point2f b, Point2f c) {
	//This function returns the coordinates of the pixels covering, at least partially, the triangle
	//First check that the triangle is no non-degenerate: area test

	//vector<pair<int, int>> coordinates;
	//if ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x) == 0) return coordinates; //maybe test the area for less than a threshhold?
	
	Point2f small, center, big;
	if (a.x <= b.x) {
		if (a.x <= c.x) {
			if (b.x <= c.x) {
				small = a;
				center = b;
				big = c;
			}
			else {
				small = a;
				center = c;
				big = b;
			}
		}	
		else {
			small = c;
			center = a;
			big = b;
		}
	}
	else {
		if (a.x <= c.x) {
			small = b;
			center = a;
			big = c;
		}
		else {
			if(b.x <= c.x) {
				small = b;
				center = c;
				big = a;
			}
			else {
				small = c;
				center = b;
				big = a;
			}
		}
	}

	//cout << endl;
	//cout << "small to big" << endl;
	vector<pair<int, int>> small_to_big = bresenham2((int)small.x, (int)small.y, (int)big.x, (int)big.y);
	if (small_to_big.front().first != (int)small.x) reverse(small_to_big.begin(), small_to_big.end());
	//cout << "small_to_big size " << small_to_big.size() << endl << endl;

	//cout << "small to center" << endl;
	vector<pair<int, int>> small_to_center = bresenham2((int)small.x, (int)small.y, (int)center.x, (int)center.y);
	if (small_to_center.front().first != (int)small.x) reverse(small_to_big.begin(), small_to_big.end());
	//cout << "small_to_center size " << small_to_center.size() << endl << endl;

	//cout << "center to big" << endl;
	vector<pair<int, int>> center_to_big = bresenham2((int)center.x, (int)center.y, (int)big.x, (int)big.y);
	if (center_to_big.front().first != (int)center.x) reverse(center_to_big.begin(), center_to_big.end());
	//cout << "center_to_big size " << center_to_big.size() << endl << endl;
	
	//If small_to_big lies perfectly under center_to_big
	int new_x;
	int new_y;

	if (small_to_big.front().first == center_to_big.front().first) {
		//cout << "small_to_big.front().first == center_to_big.front().first" << endl;
		int runner1 = 0;
		int runner2 = 0;
		int i = small_to_big.front().first;
		int end_cond = small_to_big.back().first;
		do {
			//cout << i << endl;
			if (small_to_big.at(runner1).second < center_to_big.at(runner2).second) {
				//cout << "if (small_to_big.at(runner1).second < center_to_big.at(runner2).second) {" << endl;
				int height = center_to_big.at(runner2).second - small_to_big.at(runner1).second;
				//cout << "height " << height << endl;
				for (int j = 0; j < height; ++j) {
					new_x = small_to_big.at(runner1).first;
					new_y = small_to_big.at(runner1).second + j;
					//coordinates.push_back(make_pair(new_x, new_y));
					//cout << "new coordinate (" << new_x << "," << new_y << ")" << endl;
				}
				//cout << "coordinates size " << coordinates.size() << endl;
			}
			else {
				//cout << "else" << endl;
				int height = small_to_big.at(runner1).second - center_to_big.at(runner2).second;
				//cout << "height " << height << endl;
				for (int j = 0; j < height; ++j) {
					new_x = small_to_big.at(runner1).first;
					new_y = small_to_big.at(runner1).second - j;
					//coordinates.push_back(make_pair(new_x, new_y));
					//cout << "new coordinate (" << new_x << "," << new_y << ")" << endl;
				}
				//cout << "coordinates size " << coordinates.size() << endl;
			}
			//Find next position that does not have the same value
			int small_to_big_size = small_to_big.size();
			while (runner1 < small_to_big_size) {
				runner1++;
				if (small_to_big.at(runner1).first > small_to_big.at(runner1 - 1).first) break;
			}
			int center_to_big_size = center_to_big.size();
			while (runner2 < center_to_big_size) {
				runner2++;
				if (center_to_big.at(runner2).first > center_to_big.at(runner2 - 1).first) break;
			}
			i = small_to_big.at(runner1).first;
			//cout << "runner1 " << runner1 << endl;
			//cout << "runner2 " << runner2 << endl;
		} while (i != end_cond);
		//cout << i << endl;
		int center_to_big_size = center_to_big.size();
		for (int i = 0; i < center_to_big_size; ++i) {
			//coordinates.push_back(center_to_big.at(i));
		}
	}
	else {
		//int cut = center_to_big.front().first;
	}

	/*
	cout << endl;
	for (int i = 0; i < coordinates.size(); ++i) {
		cout << "next coordinate (" << coordinates.at(i).first << "," << coordinates.at(i).second << ")" << endl;
	}
	cout << endl << "number of coordinates " << coordinates.size() << endl;
	*/

	//return coordinates;
}

void triangle_pixels2(float vec_a_x[], float vec_a_y[], float vec_b_x[], float vec_b_y[], float vec_c_x[], float vec_c_y[] ) {
}