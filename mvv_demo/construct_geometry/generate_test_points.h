#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

std::vector<cv::Point2f> get_one_sample_point();
std::vector<cv::Point2f> get_small_sample_points();
std::vector<cv::Point2f> get_sample_points();
std::vector<cv::Point2f> get_n_random_points(cv::Rect boundingBox, int n);