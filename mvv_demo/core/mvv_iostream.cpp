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

int io_test() {
	// header is 64 bytes

	
	auto t1 = std::chrono::high_resolution_clock::now();


	string imagePathA = "david_1";
	string imagePathB = "david_2";
	string rootPath = "../data_store";


	auto t3 = std::chrono::high_resolution_clock::now();
	cout << "deleting"
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1).count()
		<< " milliseconds\n";


	return 0;
}

// what IO functions should we have:

void save_csv(string folder_name, string file_name, float** affine_params) {

}

float** read_csv(string folder_name, string file_name) {
	// will read entire csv file
	// csv[triangle#][12floats]
	

	// open file
}

void save_frame(string folder_name, string file_name, char** img) {
	// this should not need to be invoked, we alraedy have the images
}

char** read_frame(string folder_name, string file_name) {
	// we will need a way to quickly read out an image to a char[][]
	// for cuda processing
}

void save_frame_t(string folder_name, string file_name, short** frame_triangle_grid) {
	// 
}

short** read_frame_t(string folder_name, string file_name) {
	short** frame = 0;
	return frame;
}