#include "mvv_iostream.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

// what IO functions should we have:

void save_csv(string folder_name, string file_name, float** affine_params, int num_triangles) {
	ofstream csv_file;
	string full_path = "../data_store/" + folder_name + "/" + file_name;
	csv_file.open(full_path);
	for (int i = 0; i < num_triangles; i++) {
		// affine parameters (12 values, 6 forward, 6 reverse)
		int num_affine_params = 12;
		for (int j = 0; j < num_affine_params; j++) {
			csv_file << affine_params[i][j] << ",";
		}
		csv_file << "\n";
	}
	csv_file.close();
}

float** read_csv(string folder_name, string file_name) {
	// will read entire csv file
	// csv[triangle#][12floats]


	// open file
	float** test = 0;
	return test;
}

void save_frame(string folder_name, string file_name, char** img) {
	// this should not need to be invoked, we alraedy have the images
}

char** read_frame(string folder_name, string file_name) {
	// we will need a way to quickly read out an image to a char[][]
	// for cuda processing

	// use imageMagick
	char** test = 0;
	return test;
}

void save_frame_t(string folder_name, string file_name, short** frame_triangle_grid) {
	// 
}

short** read_frame_t(string folder_name, string file_name) {
	short** frame = 0;
	return frame;
}

int io_test() {
	// header is 64 bytes
	int num_triangles = 300;
	int num_affine_params = 12;

	float** floats = new float*[num_triangles];
	for (int i = 0; i < num_triangles; i++) {
		floats[i] = new float[num_affine_params];
		for (int j = 0; j < num_affine_params; j++) {
			floats[i][j] = (float)i * 0.3425 * (float)j;
		}
	}

	save_csv("csv", "triangles.csv", floats, num_triangles);

	return 0;
}

