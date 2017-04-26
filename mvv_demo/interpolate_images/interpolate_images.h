#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

std::vector<std::vector<double>> interpolation_preprocessing(std::vector<cv::Vec6f> sourceT, std::vector<cv::Vec6f> targetT);

std::vector<cv::Vec6f> get_interpolated_triangles(std::vector<cv::Vec6f> sourceT, std::vector<cv::Vec6f> targetT, std::vector<std::vector<double>> affine, int tInt);

void display_interpolated_triangles(std::vector<cv::Vec6f> triangles, cv::Rect imageBounds);

void interpolation_trackbar(std::vector<cv::Vec6f> trianglesA, std::vector<cv::Vec6f> trianglesB, cv::Rect imgSizeA, cv::Rect imgSizeB, std::vector<std::vector<double>> affine);
