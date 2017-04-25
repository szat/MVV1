#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

cv::Mat test_affine_transform(cv::Vec6f sourceTri, cv::Vec6f targetTri);

std::vector<cv::Mat> get_affine_transforms(std::vector<cv::Vec6f> sourceT, std::vector<cv::Vec6f> targetT);