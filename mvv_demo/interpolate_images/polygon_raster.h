#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

std::vector<cv::Point> raster_triangle(cv::Vec6f &t, int imgWidth, int imgHeight);

std::vector<std::vector<cv::Point>> raster_triangulation(std::vector<cv::Vec6f> &triangles, cv::Rect imgBounds);

void render_rasterization(std::vector<std::vector<cv::Point>> raster, cv::Rect imgBounds);