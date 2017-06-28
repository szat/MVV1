#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

// Note about all of this:
// OS-dependent! This will not work on other architectures (for now).

void write_uchar_array(std::string full_path, char * input, int length, int width, int height) {
	std::ofstream ofile(full_path, std::ios::binary);
	char * length_array = new char[12];
	int * int_array = new int[3];
	int_array[0] = length;
	int_array[1] = width;
	int_array[2] = height;
	memcpy(length_array, int_array, 12);
	ofile.write(length_array, 12);
	ofile.write(input, length);
	ofile.close();
}

void write_short_array(std::string full_path, short * input, int length) {
	char * char_input = new char[length * 2];
	std::ofstream ofile(full_path, std::ios::binary);
	char * length_array = new char[4];
	int * int_array = new int[1];
	int_array[0] = length;
	memcpy(length_array, int_array, 4);
	ofile.write(length_array, 4);
	memcpy(char_input, input, length * 2);
	ofile.write(char_input, length * 2);
}

void write_float_array(std::string full_path, float * input, int length) {
	char * char_input = new char[length * 4];
	std::ofstream ofile(full_path, std::ios::binary);
	char * length_array = new char[4];
	int * int_array = new int[1];
	int_array[0] = length;
	memcpy(length_array, int_array, 4);
	ofile.write(length_array, 4);
	memcpy(char_input, input, length * 4);
	ofile.write(char_input, length * 4);
}

void save_raster(std::string full_path, short ** raster, int width, int height) {
	int size = width * height;
	short * raster_1D = new short[size];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			raster_1D[i * width + j] = raster[i][j];
		}
	}
	write_short_array(full_path, raster_1D, size);
}

float* convert_vector_params(std::vector<cv::Mat> forward_params, std::vector<cv::Mat> reverse_params) {
	int num_triangles = forward_params.size();
	float* params = new float[num_triangles * 12];
	for (int i = 0; i < num_triangles; i++) {
		int inc = 12 * i;
		params[inc] = (float)forward_params[i].at<double>(0, 0);
		params[inc + 1] = (float)forward_params[i].at<double>(0, 1);
		params[inc + 2] = (float)forward_params[i].at<double>(0, 2);
		params[inc + 3] = (float)forward_params[i].at<double>(1, 0);
		params[inc + 4] = (float)forward_params[i].at<double>(1, 1);
		params[inc + 5] = (float)forward_params[i].at<double>(1, 2);
		params[inc + 6] = (float)reverse_params[i].at<double>(0, 0);
		params[inc + 7] = (float)reverse_params[i].at<double>(0, 1);
		params[inc + 8] = (float)reverse_params[i].at<double>(0, 2);
		params[inc + 9] = (float)reverse_params[i].at<double>(1, 0);
		params[inc + 10] = (float)reverse_params[i].at<double>(1, 1);
		params[inc + 11] = (float)reverse_params[i].at<double>(1, 2);
	}
	return params;
}

void save_img(std::string tar_path, cv::Mat &img) {
	cv::Size size = img.size();
	int height = size.height;
	int width = size.width;
	int len = height * width * 3;

	uchar *pixels = new uchar[len];

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cv::Vec3b data = img.at<cv::Vec3b>(i, j);
			int index = (i * width + j) * 3;
			pixels[index] = data[0];
			pixels[index + 1] = data[1];
			pixels[index + 2] = data[2];
		}
	}

	char *char_result = new char[len];
	memcpy(char_result, pixels, len);
	write_uchar_array(tar_path, char_result, len, width, height);
}


// this next function will be to encode the images in a binary format
// this will use openCV because we don't care about speed in the encoding (only the decoding, during the interpolation step).
void save_img_binary(cv::Mat next_1, cv::Mat next_2, cv::Size desired_size, std::string imgA_path, std::string imgB_path) {
	//cv::Mat img_1 = imread(src_path_1, cv::ImreadModes::IMREAD_COLOR);
	//cv::Mat img_2 = imread(src_path_2, cv::ImreadModes::IMREAD_COLOR);
	//cv::resize(next_1, next_1, desired_size);
	//cv::resize(next_2, next_2, desired_size);

	save_img(imgA_path, next_1);
	save_img(imgB_path, next_2);
}