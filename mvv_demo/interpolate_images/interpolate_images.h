#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

std::vector<cv::Mat> get_affine_transforms(std::vector<cv::Vec6f> sourceT, std::vector<cv::Vec6f> targetT);

std::vector<cv::Vec6f> get_interpolated_triangles(std::vector<cv::Vec6f> sourceT, std::vector<cv::Vec6f> targetT, std::vector<std::vector<std::vector<double>>> affine, int tInt);

void display_interpolated_triangles(std::vector<cv::Vec6f> triangles, cv::Rect imageBounds);

cv::Mat get_affine_intermediate(cv::Mat affine, float t);

void interpolation_trackbar(std::vector<cv::Vec6f> trianglesA, std::vector<cv::Vec6f> trianglesB, cv::Rect imgSizeA, cv::Rect imgSizeB, std::vector<std::vector<std::vector<double>>> affine);

void purple_mesh_test();

void save_frame_at_tau(cv::Mat &imgA, cv::Mat &imgB, cv::Rect imgRect, std::vector<cv::Mat> affineForward, std::vector<cv::Mat> affineReverse, std::vector<cv::Vec6f> trianglesA, std::vector<cv::Vec6f> trianglesB, float tau);

std::vector<cv::Point> fill_triangle(cv::Vec6f triangle);