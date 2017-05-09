#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <ctime>
#include <iostream>
#include "polygon_raster.h"

using namespace cv;
using namespace std;

int orient2d(const Point& a, const Point& b, const Point& c)
{
	return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

int min3(int x, int y, int z) {
	int temp = min(x, y);
	int minimum = min(temp, z);
	return minimum;
}

int max3(int x, int y, int z) {
	int temp = max(x, y);
	int maximum = max(temp, z);
	return maximum;
}

// Replace with more efficient method
vector<Point> raster_triangle(Vec6f &t, int imgWidth, int imgHeight)
{
	vector<Point> points = vector<Point>();
	Point v0 = Point(t[0], t[1]);
	Point v1 = Point(t[2], t[3]);
	Point v2 = Point(t[4], t[5]);
	// Compute triangle bounding box
	int minX = min3(v0.x, v1.x, v2.x);
	int minY = min3(v0.y, v1.y, v2.y);
	int maxX = max3(v0.x, v1.x, v2.x);
	int maxY = max3(v0.y, v1.y, v2.y);

	// Clip against screen bounds
	minX = max(minX, 0);
	minY = max(minY, 0);
	maxX = min(maxX, imgWidth - 1);
	maxY = min(maxY, imgHeight - 1);

	// Rasterize
	Point p;
	for (p.y = minY; p.y <= maxY; p.y++) {
		for (p.x = minX; p.x <= maxX; p.x++) {
			// Determine barycentric coordinates

			if (p.x == 80 && p.y == 600) {
				int i = 0;
			}

			int w0 = orient2d(v1, v2, p);
			int w1 = orient2d(v2, v0, p);
			int w2 = orient2d(v0, v1, p);

			// If p is on or inside all edges, render pixel.
			if (w0 >= 0 && w1 >= 0 && w2 >= 0)
				points.push_back(p);
		}
	}
	return points;
}

vector<vector<Point>> raster_triangulation(vector<Vec6f> &triangles, Rect imgBounds) {
	vector<vector<Point>> raster = vector<vector<Point>>();
	int size = triangles.size();
	float width = imgBounds.width;
	float height = imgBounds.height;
	for (int i = 0; i < size; i++) {
		vector<Point> rasteredTriangle = raster_triangle(triangles[i], width, height);
		raster.push_back(rasteredTriangle);
	}
	return raster;
}

vector<int> get_colors(int colorCase) {
	vector<int> colors = vector<int>();

	if (colorCase == 0) {
		colors.push_back(255);
		colors.push_back(0);
		colors.push_back(0);
	}
	else if (colorCase == 1) {
		colors.push_back(0);
		colors.push_back(255);
		colors.push_back(0);
	}
	else if (colorCase == 2) {
		colors.push_back(0);
		colors.push_back(0);
		colors.push_back(255);
	}
	else if (colorCase == 3) {
		colors.push_back(140);
		colors.push_back(0);
		colors.push_back(0);
	}
	else if (colorCase == 4) {
		colors.push_back(0);
		colors.push_back(140);
		colors.push_back(0);
	}
	else if (colorCase == 5) {
		colors.push_back(0);
		colors.push_back(0);
		colors.push_back(140);
	}
	else {
		throw "Color case error";
	}
	return colors;
}

void render_rasterization(vector<vector<Point>> raster, Rect imgBounds) {
	int width = imgBounds.width;
	int height = imgBounds.height;
	Mat img(height, width, CV_8UC3, Scalar(0, 0, 0));
	int numTriangles = raster.size();

	for (int i = 0; i < numTriangles; i++) {
		int numPixels = raster[i].size();
		for (int j = 0; j < numPixels; j++) {
			Point pixel = raster[i][j];
			int colorCase = i % 6;
			vector<int> colors = get_colors(colorCase);
			img.at<cv::Vec3b>(pixel.y, pixel.x)[0] = colors[0];
			img.at<cv::Vec3b>(pixel.y, pixel.x)[1] = colors[1];
			img.at<cv::Vec3b>(pixel.y, pixel.x)[2] = colors[2];
		}
	}

	imshow("raster", img);
	waitKey(1);
}

int** grid_from_raster(int width, int height, vector<vector<Point>> raster) {
	int** grid = new int*[height + 1];
	// Initializing arrays to default -1 value, which indicates no triangulation in this region.

	for (int h = 0; h < height + 1; h++)
	{
		grid[h] = new int[width + 1];
		for (int w = 0; w < width + 1; w++)
		{
			grid[h][w] = -1;
		}
	}

	int numRaster = raster.size();

	for (int i = 0; i < numRaster; i++) {
		int numPixels = raster[i].size();
		for (int j = 0; j < numPixels; j++) {
			int x = raster[i][j].x;
			int y = raster[i][j].y;
			// weird index swapping
			// triangle index for the affine transforms
			grid[y][x] = i;
		}
	}
	// possible memory leak if this gets called a bunch and the arrays are never deleted
	return grid;
}

void check_grid(int** grid, Rect imgBounds) {
	int width = imgBounds.width;
	int height = imgBounds.height;
	int count = 0;
	int invalidCount = 0;
	for (int h = 0; h < height + 1; h++)
	{
		for (int w = 0; w < width + 1; w++)
		{
			if (grid[h][w] == -1) {
				invalidCount++;
			}
			count++;
		}
	}
	cout << "Invalid count: " << invalidCount << endl;
	cout << "Total count: " << count << endl;
}