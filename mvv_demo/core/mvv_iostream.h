#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

float* convert_vector_params(std::vector<cv::Mat> forward_params, std::vector<cv::Mat> reverse_params);
