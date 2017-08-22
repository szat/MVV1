#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

std::vector<cv::Mat> get_affine_transforms_forward(const std::vector<cv::Vec6f> &triangle_list_1, const std::vector<cv::Vec6f> &triangle_list_2);

std::vector<cv::Mat> get_affine_transforms_reverse(const std::vector<cv::Vec6f> &triangle_list_1, const std::vector<cv::Vec6f> &triangle_list_2, const std::vector<cv::Mat> &forward_transforms);

std::vector<cv::Vec6f> get_interpolated_triangles(const std::vector<cv::Vec6f> &triangle_list_1, const std::vector<std::vector<std::vector<double>>> &affine, const int tInt);

void display_interpolated_triangles(const std::vector<cv::Vec6f> & triangles, const cv::Rect & image_bounds);

void reverse_transform(const cv::Mat &forward, cv::Mat &reverse);

void interpolation_trackbar(const std::vector<cv::Vec6f> & triangle_list_1, const  std::vector<cv::Vec6f> & triangle_list_2, const cv::Rect & img1_size, const cv::Rect & img2_size, const std::vector<std::vector<std::vector<double>>> & affine);