#pragma once

#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

uchar * read_uchar_array(std::string full_path, int &length, int &width, int &height);

void save_raster(std::string full_path, short ** raster, int width, int height);

void write_float_array(std::string full_path, float * input, int length);

float* convert_vector_params(std::vector<cv::Mat> forward_params, std::vector<cv::Mat> reverse_params);

void save_img(std::string tar_path, cv::Mat &img);

void save_img_binary(std::string src_path_1, std::string tar_path_1, std::string src_path_2, std::string tar_path_2);