#pragma once

#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

void save_raster(std::string full_path, short ** raster, int width, int height);

void write_float_array(std::string full_path, float * input, int length);

float* convert_vector_params(std::vector<cv::Mat> forward_params, std::vector<cv::Mat> reverse_params);