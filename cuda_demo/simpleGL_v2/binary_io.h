#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

short * read_short_array(std::string full_path, int &length) {
	std::ifstream ifile(full_path, std::ios::binary);
	char * int_array = new char[4];
	ifile.read(int_array, 4);
	for (int k = 0; k < 4; k++) {
		unsigned char len_char = (unsigned char)int_array[k];
		length += len_char << k * 8;
	}
	char * result = new char[length * 2];
	ifile.read(result, length * 2);
	short * short_result = new short[length];
	memcpy(short_result, result, length * 2);
	return short_result;
}

float * read_float_array(std::string full_path, int &length) {
	std::ifstream ifile(full_path, std::ios::binary);
	char * int_array = new char[4];
	ifile.read(int_array, 4);
	for (int k = 0; k < 4; k++) {
		unsigned char len_char = (unsigned char)int_array[k];
		length += len_char << k * 8;
	}
	float * result = new float[length];
	char * char_result = new char[length * 4];
	ifile.read(char_result, length * 4);
	memcpy(result, char_result, length * 4);
	return result;
}