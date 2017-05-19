#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

uchar * read_uchar_array(std::string full_path, int &length, int &width, int &height) {
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
	uchar * result_uchar = new uchar[length];
	memcpy(result_uchar, result, length);
	return result_uchar;
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
	short * short_result = new short[length];
	memcpy(short_result, result, length * 2);
	return short_result;
}

float * read_float_array(std::string full_path, int &length) {
	std::ifstream ifile(full_path, std::ios::binary);
	char * length_array = new char[4];
	int * int_array = new int[1];
	ifile.read(length_array, 4);
	memcpy(int_array, length_array, 4);
	length = int_array[0];
	float * result = new float[length];
	char * char_result = new char[length * 4];
	ifile.read(char_result, length * 4);
	memcpy(result, char_result, length * 4);
	return result;
}