#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>


uchar3 * read_uchar3_array(std::string full_path, int &length, int &width, int &height) {
	// modifies length, returns char array
	std::ifstream ifile(full_path, std::ios::binary);
	char * length_array = new char[12];
	int * int_array = new int[3];
	ifile.read(length_array, 12);
	memcpy(int_array, length_array, 12);
	length = int_array[0];
	width = int_array[1];
	height = int_array[2];
	char * result = new char[length];
	ifile.read(result, length);
	uchar3 * result_uchar3 = reinterpret_cast<uchar3*>(result);
	result = nullptr;
	free(length_array);
	free(int_array);
	ifile.close();
	return result_uchar3;
}

short * read_short_array(std::string full_path, int &length) {
	std::ifstream ifile(full_path, std::ios::binary);
	char * length_array = new char[4];
	int * int_array = new int[1];
	ifile.read(length_array, 4);
	memcpy(int_array, length_array, 4);
	length = int_array[0];
	char * result = new char[length * 2];
	ifile.read(result, length * 2);
	short * short_result = reinterpret_cast<short*>(result);
	result = nullptr;
	free(length_array);
	free(int_array);
	ifile.close();
	return short_result;
}

float * read_float_array(std::string full_path, int &length) {
	std::ifstream ifile(full_path, std::ios::binary);
	char * length_array = new char[4];
	int * int_array = new int[1];
	ifile.read(length_array, 4);
	memcpy(int_array, length_array, 4);
	length = int_array[0];
	char * char_result = new char[length * 4];
	ifile.read(char_result, length * 4);
	float * float_result = reinterpret_cast<float*>(char_result);
	char_result = nullptr;
	free(length_array);
	free(int_array);
	ifile.close();
	return float_result;
}