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
using namespace cv::ximgproc;

int main()
{
	cout << "Welcome to spx_demo, code to try out the SLIC segmentation method!" << endl;
	cout << "Press any key to exit." << endl;

	
	string img_file = "..\\data_store\\images\\mona_lisa_1.jpg";

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

	Mat converted;
	cvtColor(input_image, converted, COLOR_BGR2HSV);
	Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(converted, algorithm + SLIC, region_size, float(ruler));
	slic->iterate(num_iterations);
	if (min_element_size > 0) {
		slic->enforceLabelConnectivity(min_element_size);
	}
	Mat result, mask;
	result = input_image;
	slic->getLabelContourMask(mask, true);
	result.setTo(Scalar(0, 0, 255), mask);

	Mat labels;
	slic->getLabels(labels);
	const int num_label_bits = 2;
	labels &= (1 << num_label_bits) - 1;
	labels *= 1 << (16 - num_label_bits);

	string window_name_r = "SLIC result";
	//string window_name_m = "SLIC mask";
	//string window_name_l = "SLIC label";
	
	namedWindow(window_name_r);
	//namedWindow(window_name_m);
	//namedWindow(window_name_l);

	imshow(window_name_r, result);
	//imshow(window_name_m, mask);
	//imshow(window_name_l, labels);

	waitKey();

    return 0;
}

