#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace std;
using namespace cv;

vector<Point2f> get_one_sample_point();
vector<Point2f> get_small_sample_points();
vector<Point2f> get_sample_points();
vector<Point2f> get_n_random_points(Rect boundingBox, int n);