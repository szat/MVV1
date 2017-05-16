#include "binary_io.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;
using namespace cv;
// Note about all of this:
// OS-dependent! This will not work on other architectures (for now).


void write_char_array(string full_path, char * input, int length) {
	ofstream ofile(full_path, ios::binary);
	char * length_array = new char[4];
	for (int k = 0; k < 4; k++) {
		length_array[k] = (length >> k * 8) & 0xFF;
	}
	ofile.write(length_array, 4);
	ofile.write(input, length);
	ofile.close();
}

char * read_char_array(string full_path, int &length) {
	// modifies length, returns char array
	ifstream ifile(full_path, ios::binary);
	char * int_array = new char[4];
	ifile.read(int_array, 4);
	for (int k = 0; k < 4; k++) {
		unsigned char len_char = (unsigned char)int_array[k];
		length += len_char << k * 8;
	}
	char * result = new char[length];
	ifile.read(result, length);
	return result;
}

void write_short_array(string full_path, short * input, int length) {
	char * char_input = new char[length * 2];
	ofstream ofile(full_path, ios::binary);
	char * length_array = new char[4];
	for (int k = 0; k < 4; k++) {
		length_array[k] = (length >> k * 8) & 0xFF;
	}
	memcpy(char_input, input, length * 2);
	ofile.write(length_array, 4);
	ofile.write(char_input, length * 2);
}

short * read_short_array(string full_path, int &length) {
	ifstream ifile(full_path, ios::binary);
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

void write_float_array(string full_path, float * input, int length) {
	char * char_input = new char[length * 4];
	ofstream ofile(full_path, ios::binary);
	char * length_array = new char[4];
	for (int k = 0; k < 4; k++) {
		length_array[k] = (length >> k * 8) & 0xFF;
	}
	ofile.write(length_array, 4);
	memcpy(char_input, input, length * 4);
	ofile.write(char_input, length * 4);
}

float * read_float_array(string full_path, int &length) {
	ifstream ifile(full_path, ios::binary);
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

void timing() {
	int len = 1000000;

	short * stuff = new short[len];
	for (int i = 0; i < len; i++) {
		stuff[i] = (short)i;
	}

	char * stuff2 = new char[len];
	for (int i = 0; i < len; i++) {
		stuff2[i] = (char)i;
	}

	auto t1 = std::chrono::high_resolution_clock::now();


	string full_path = "../data_store/foobar.bin";
	write_short_array(full_path, stuff, len);

	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "write short took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
		<< " milliseconds\n";

	int length = 0;
	short *result = read_short_array(full_path, length);

	auto t3 = std::chrono::high_resolution_clock::now();
	std::cout << "read short took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
		<< " milliseconds\n";

	string full_path2 = "../data_store/foobar2.bin";
	write_char_array(full_path2, stuff2, len);

	auto t4 = std::chrono::high_resolution_clock::now();
	std::cout << "write char took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
		<< " milliseconds\n";

	int length2 = 0;
	read_char_array(full_path2, length2);

	auto t5 = std::chrono::high_resolution_clock::now();
	std::cout << "read char took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
		<< " milliseconds\n";
}

void save_raster(string full_path, short ** raster, int width, int height) {
	int size = width * height;
	short * raster_1D = new short[size];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			raster_1D[i * width + j] = raster[i][j];
		}
	}
	write_short_array(full_path, raster_1D, size);
}

float* convert_vector_params(vector<Mat> forward_params, vector<Mat> reverse_params) {
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


void test_binary() {
	cout << "Test binary";
	timing();
	cout << "Testing";
	// test time for 1 million char, 1 million short
};