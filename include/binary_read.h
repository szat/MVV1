#pragma once
#include <chrono>
#include <string>
#include <stdio.h>
#include <stdlib.h>

uchar3 * read_uchar3_array(std::string full_path, int &length, int &width, int &height) {
	// modifies length, returns char array
	const char *file_path = full_path.c_str();
	FILE *f = fopen(file_path, "rb");
	char * length_array = new char[12];
	int * int_array = new int[3];

	fread(length_array, 12, 1, f);
	memcpy(int_array, length_array, 12);

	length = int_array[0];
	width = int_array[1];
	height = int_array[2];

	char * result = new char[length];
	fread(result, length, 1, f);

	uchar3 * result_uchar3 = reinterpret_cast<uchar3*>(result);

	result = nullptr;
	free(length_array);
	free(int_array);
	fclose(f);
	return result_uchar3;
}

short * read_short_array(std::string full_path, int &length) {
	const char *file_path = full_path.c_str();
	FILE *f = fopen(file_path, "rb");
	char * length_array = new char[4];
	int * int_array = new int[1];
	fread(length_array, 4, 1, f);
	memcpy(int_array, length_array, 4);
	length = int_array[0];
	char * result = new char[length * 2];
	fread(result, length * 2, 1, f);
	short * short_result = reinterpret_cast<short*>(result);
	result = nullptr;
	free(length_array);
	free(int_array);
	fclose(f);
	return short_result;
}

float * read_float_array(std::string full_path, int &length) {
	const char *file_path = full_path.c_str();
	FILE *f = fopen(file_path, "rb");
	char * length_array = new char[4];
	int * int_array = new int[1];
	fread(length_array, 4, 1, f);
	memcpy(int_array, length_array, 4);
	length = int_array[0];
	char * char_result = new char[length * 4];
	fread(char_result, length * 4, 1, f);
	float * float_result = reinterpret_cast<float*>(char_result);
	char_result = nullptr;
	free(length_array);
	free(int_array);
	fclose(f);
	return float_result;
}