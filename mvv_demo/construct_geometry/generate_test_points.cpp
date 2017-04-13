#include "stdafx.h"
#include "generate_test_points.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <random>
#include <ctime>

using namespace std;
using namespace cv;

// get_one_sample_point and get_small_sample_points and get_sample_points
// are good static test cases. Please leave them in the code.

std::vector<Point2f> get_one_sample_point() {
	vector<Point2f> points = vector<Point2f>();
	points.push_back(Point2f(500, 20));
	return points;
}

std::vector<Point2f> get_small_sample_points() {
	vector<Point2f> points = vector<Point2f>();
	points.push_back(Point2f(528, 390));
	points.push_back(Point2f(551, 209));
	points.push_back(Point2f(289, 355));
	points.push_back(Point2f(569, 105));
	points.push_back(Point2f(155, 551));
	return points;
}

std::vector<Point2f> get_sample_points() {
	vector<Point2f> points = vector<Point2f>();
	points.push_back(Point2f(528, 390));
	points.push_back(Point2f(551, 209));
	points.push_back(Point2f(289, 355));
	points.push_back(Point2f(569, 105));
	points.push_back(Point2f(155, 551));
	points.push_back(Point2f(335, 319));
	points.push_back(Point2f(273, 331));
	points.push_back(Point2f(353, 505));
	points.push_back(Point2f(590, 287));
	points.push_back(Point2f(305, 31));
	points.push_back(Point2f(25, 313));
	points.push_back(Point2f(16, 374));
	points.push_back(Point2f(44, 57));
	points.push_back(Point2f(517, 335));
	points.push_back(Point2f(564, 475));
	points.push_back(Point2f(508, 548));
	points.push_back(Point2f(126, 494));
	points.push_back(Point2f(553, 267));
	points.push_back(Point2f(52, 565));
	points.push_back(Point2f(91, 589));
	points.push_back(Point2f(234, 478));
	points.push_back(Point2f(179, 211));
	points.push_back(Point2f(261, 324));
	points.push_back(Point2f(392, 91));
	points.push_back(Point2f(560, 592));
	points.push_back(Point2f(578, 383));
	points.push_back(Point2f(260, 316));
	points.push_back(Point2f(529, 14));
	points.push_back(Point2f(70, 507));
	points.push_back(Point2f(264, 477));
	points.push_back(Point2f(188, 119));
	points.push_back(Point2f(156, 125));
	points.push_back(Point2f(207, 251));
	points.push_back(Point2f(162, 214));
	points.push_back(Point2f(218, 592));
	points.push_back(Point2f(504, 38));
	points.push_back(Point2f(319, 267));
	points.push_back(Point2f(21, 31));
	points.push_back(Point2f(106, 88));
	points.push_back(Point2f(508, 158));
	points.push_back(Point2f(482, 158));
	points.push_back(Point2f(419, 335));
	points.push_back(Point2f(332, 554));
	points.push_back(Point2f(538, 589));
	points.push_back(Point2f(133, 198));
	points.push_back(Point2f(160, 9));
	points.push_back(Point2f(436, 462));
	points.push_back(Point2f(256, 442));
	points.push_back(Point2f(159, 130));
	points.push_back(Point2f(97, 150));
	return points;
}

std::vector<Point2f> get_n_random_points(Rect boundingBox, int n) {
	vector<Point2f> random_points = vector<Point2f>();

	if (n < 0) {
		// Do some logging to indicate that this is a fatal error.
		exit(-1);
	}

	int width = boundingBox.width;
	int height = boundingBox.height;

	srand(time(0));
	// Can be easily changed if we want a box offset from the origin (0,0)
	int xMin = 0;
	int xMax = width;
	int yMin = 0;
	int yMax = height;

	int random_x = -1;
	int random_y = -1;

	for (size_t i = 0; i < n; i++)
	{
		random_x = xMin + rand() % ((xMax - xMin));
		random_y = yMin + rand() % ((yMax - yMin));
		random_points.push_back(Point2f(random_x, random_y));
	}

	return random_points;
}