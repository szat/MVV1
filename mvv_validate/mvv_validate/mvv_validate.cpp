// mvv_validate.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <fstream>
// vector_types is from CUDA, we need it for uchar3 and uchar4 formats
// although they are trivial structs, it needs to accord with the rest of CUDA
#include <fake_types.h>
#include "binary_read.h"

using namespace std;
using namespace cv;


long get_file_size(std::string filename)
{
	struct stat stat_buf;
	int rc = stat(filename.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : -1;
}

void test_image(string filename, string directory) {
	bool success = true;
	int header_offset = 12;
	string full_path = directory + filename;

	long file_size = get_file_size(full_path);

	int length = 0;
	int width = 0;
	int height = 0;
	uchar3 * result = read_uchar3_array(full_path, length, width, height);

	if (length == width * height * 3) {
		cout << "Binary body length corresponds to what we would expect" << endl;
	}
	else {
		cout << "Error: Binary body length does not correspond to desired outcome" << endl;
		success = false;
	}

	if (file_size == width * height * 3 + header_offset) {
		cout << "Binary file length corresponds to what we would expect" << endl;
	}
	else {
		cout << "Error: Binary file length does not correspond to desired outcome" << endl;
		success = false;
	}

	cout << "Length: " << length << " Width: " << width << " Height: " << height << endl;
	if (success) {
		cout << "File is valid" << endl;
	}
	else {
		cout << "File is invalid" << endl;
	}
}

short * test_raster(string filename, string directory, int expected_width, int expected_height) {
	bool success = true;
	int header_offset = 4;
	string full_path = directory + filename;

	long file_size = get_file_size(full_path);

	int length = 0;
	short * result = read_short_array(full_path, length);

	if (length == expected_width * expected_height ) {
		cout << "Binary body length corresponds to what we would expect" << endl;
	}
	else {
		cout << "Error: Binary body length does not correspond to desired outcome" << endl;
		success = false;
	}

	int expected_file_size = expected_width * expected_height * 2 + header_offset;
	if (file_size == expected_file_size) {
		cout << "Binary file length corresponds to what we would expect" << endl;
	}
	else {
		cout << "Error: Binary file length does not correspond to desired outcome" << endl;
		success = false;
	}

	if (success) {
		cout << "File is valid" << endl;
	}
	else {
		cout << "File is invalid" << endl;
	}
	return result;
}

void test_affine(string affine_directory, string file_path, short *raster_A, short *raster_B, int width, int height) {
	int length = 0;
	string full_path = affine_directory + file_path;
	float * affine_params = read_float_array(full_path, length);

	bool valid = true;

	int num_triangles = length / 12;

	if (length % 12 != 0) {
		cout << "Invalid number of floats";
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int triangle = raster_A[i * width + j];
			int offset = 0;
			float f_x = (float)j;
			float f_y = (float)i;
			float a_00 = affine_params[triangle * 12 + offset];
			float a_01 = affine_params[triangle * 12 + offset + 1];
			float b_00 = affine_params[triangle * 12 + offset + 2];
			float a_10 = affine_params[triangle * 12 + offset + 3];
			float a_11 = affine_params[triangle * 12 + offset + 4];
			float b_11 = affine_params[triangle * 12 + offset + 5];
			float final_j = a_00 * f_x + a_01 * f_y + b_00;
			float final_i = a_10 * f_y + a_11 * f_y + b_11;
			int final_i_int = (int)final_i;
			int final_j_int = (int)final_j;

			if (final_i_int < 0 || final_i_int > height || final_j_int < 0 || final_j_int > width) {
				valid = false;
				cout << "Out of bounds, i = " << final_i_int << ", j = " << final_j_int << endl;
			}
		}
	}


}


int main()
{
	// The purpose of the mvv_validate .exe is to:
	// -validate .mvv/.bin files generated by mvv_demo
	// -render these files using openCV

	cout << "Beginning mvv validate" << endl;
	

	string binary_directory = "..\\..\\data_store\\binary\\";
	string david_1_filename = "david_1.bin";
	string david_2_filename = "david_2.bin";
	string judo_1_filename = "judo_1.bin";
	string judo_2_filename = "judo_2.bin";

	// One part of the algorithm that is suspect is the saving of the images in binary format. We will invoke
	// test_image on david and the judo shot.

	test_image(david_1_filename, binary_directory);
	test_image(david_2_filename, binary_directory);
	test_image(judo_1_filename, binary_directory);
	test_image(judo_2_filename, binary_directory);
	
	// in addition to binary checks (right number of bytes), we should have an opencv render
	// of the images and of the raster

	// we should also check the affine transforms

	string raster_directory = "..\\..\\data_store\\raster\\";
	string david_1_raster_filename = "rasterA_david.bin";
	string david_2_raster_filename = "rasterB_david.bin";
	string judo_1_raster_filename = "rasterA_judo.bin";
	string judo_2_raster_filename = "rasterB_judo.bin";

	int david_width = 667;
	int david_height = 1000;
	int judo_width = 1035;
	int judo_height = 780;

	short *david_1_raster = test_raster(david_1_raster_filename, raster_directory, david_width, david_height);
	short *david_2_raster = test_raster(david_2_raster_filename, raster_directory, david_width, david_height);
	short *judo_1_raster = test_raster(judo_1_raster_filename, raster_directory, judo_width, judo_height);
	short *judo_2_raster = test_raster(judo_2_raster_filename, raster_directory, judo_width, judo_height);

	// making sure none of the affine transformations map off the page
	// checking if there are any that are zero (although this should not cause a fatal 
	// exception, it has worked before with this)


	string affine_directory = "..\\..\\data_store\\affine\\";
	string affine_david = "affine_david";
	string affine_judo = "affine_judo";
	test_affine(affine_directory, affine_david, david_1_raster, david_2_raster, david_width, david_height);
	//test_affine(affine_directory, affine_judo, judo_1_raster, judo_2_raster, judo_width, judo_height);

	cin.get();

    return 0;
}

