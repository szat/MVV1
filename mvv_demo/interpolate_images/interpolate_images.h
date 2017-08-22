#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

std::vector<cv::Mat> get_affine_transforms_forward(std::vector<cv::Vec6f> &triangle_list_1, std::vector<cv::Vec6f> &triangle_list_2);

std::vector<cv::Mat> get_affine_transforms_reverse(std::vector<cv::Vec6f> &triangle_list_1, std::vector<cv::Vec6f> &triangle_list_2, std::vector<cv::Mat> &forward_transforms);

std::vector<cv::Vec6f> get_interpolated_triangles(std::vector<cv::Vec6f> &triangle_list_1, std::vector<std::vector<std::vector<double>>> &affine, int tInt);

void display_interpolated_triangles(std::vector<cv::Vec6f> & triangles, cv::Rect & image_bounds);

void interpolation_trackbar(std::vector<cv::Vec6f> & triangle_list_1, std::vector<cv::Vec6f> & triangle_list_2, cv::Rect & img1_size, cv::Rect & img2_size, std::vector<std::vector<std::vector<double>>> & affine);