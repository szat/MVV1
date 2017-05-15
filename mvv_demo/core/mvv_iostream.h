#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

double** convert_vector_params(std::vector<cv::Mat> forward_params, std::vector<cv::Mat> reverse_params);

void save_csv(std::string folder_name, std::string file_name, double** affine_params, int num_triangles);

float** read_csv(std::string folder_name, std::string file_name);

void save_grayscale_t(std::string folder_name, std::string file_name, short** frame_triangle_grid, int width, int height);

short** read_grayscale_t(std::string folder_name, std::string file_name);

int io_test();

