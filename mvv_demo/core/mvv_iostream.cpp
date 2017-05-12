#include "mvv_iostream.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

#define ROOT_DATA_FOLDER "../data_store"

// what IO functions should we have:

void save_csv(string folder_name, string file_name, float** affine_params) {
	for (int i = 0; i < 300; i++) {
		for (int j = 0; j < 12; j++) {
			if (j != 11) {
				cout << affine_params[i][j] << ",";
			}
			else {
				cout << affine_params[i][j];
			}

		}
		cout << "\n";
	}
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

	
	float** floats = new float*[300];
	for (int i = 0; i < 300; i++) {
		floats[i] = new float[12];
		for (int j = 0; j < 12; j++) {
			floats[i][j] = (float)i * 0.3425 * (float)j;
		}
	}

	save_csv("", "", floats);

	return 0;
}

