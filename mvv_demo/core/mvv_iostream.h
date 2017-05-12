#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

float** convert_vector_params(std::vector<cv::Mat> forward_params, std::vector<cv::Mat> reverse_params);

void save_csv(std::string folder_name, std::string file_name, float** affine_params, int num_triangles);

float** read_csv(std::string folder_name, std::string file_name);

void save_frame(std::string folder_name, std::string file_name, char** img);

char** read_frame(std::string folder_name, std::string file_name);

void save_frame_t(std::string folder_name, std::string file_name, short** frame_triangle_grid);

short** read_frame_t(std::string folder_name, std::string file_name);

int io_test();

