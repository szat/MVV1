// spx_demo.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main()
{
	cout << "Welcome to spx_demo, code to try out the SLIC segmentation method!" << endl;
	cout << "Press any key to exit." << endl;

	string window_name = "SLIC Superpixels";
	string img_file = "..\\data_store\\images\\mona_lisa_1.jpg.jpg";

	Mat input_image;

	input_image = imread(img_file);
	if (input_image.empty())
	{
		cout << "Could not open image..." << img_file << "\n";
		return -1;
	}

	cout << "Passed the loading!" << endl;

	int algorithm = 0;
	int region_size = 50;
	int ruler = 30;
	int min_element_size = 50;
	int num_iterations = 3;

	namedWindow(window_name, 0);
	createTrackbar("Algorithm", window_name, &algorithm, 2, 0);
	createTrackbar("Region size", window_name, &region_size, 200, 0);
	createTrackbar("Ruler", window_name, &ruler, 100, 0);
	createTrackbar("Connectivity", window_name, &min_element_size, 100, 0);
	createTrackbar("Iterations", window_name, &num_iterations, 12, 0);

	cin.ignore();
    return 0;
}

