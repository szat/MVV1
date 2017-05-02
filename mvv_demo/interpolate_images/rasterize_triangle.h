#pragma once
#include <vector>
#include <opencv2\core\core.hpp>

std::vector<std::pair<int, int>> bresenham2(int x1, int y1, int x2, int y2);

std::vector<std::pair<int, int>> bresenham(int x0, int y0, int x1, int y1);

void triangle_pixels(cv::Point2f a, cv::Point2f b, cv::Point2f c);

void triangle_pixels2(float vec_a_x[], float vec_a_y[], float vec_b_x[], float vec_b_y[], float vec_c_x[], float vec_c_y[]);