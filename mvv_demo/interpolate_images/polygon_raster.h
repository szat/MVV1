// Created by Adrian Szatmari and Daniel Hogg, 2017
// MVV is released under the MIT license
// https://github.com/danielhogg/mvv
// https://github.com/szat/mvv

#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

std::vector<cv::Point> raster_triangle(const cv::Vec6f &t, const int img_width, const int img_height);

std::vector<std::vector<cv::Point>> raster_triangulation(const std::vector<cv::Vec6f> &triangles, const cv::Rect & imgBounds);

void render_rasterization(const std::vector<std::vector<cv::Point>> & raster, const cv::Rect & imgBounds);

short** grid_from_raster(const int width, const int height, const std::vector<std::vector<cv::Point>> & raster);