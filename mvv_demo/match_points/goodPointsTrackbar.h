#pragma once

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "goodPointsTrackbar.h"
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;

struct myData {
	Mat variable1;
	Mat variable2;
};

static void onChange(int trackpos, void *userdata) //void* mean that it is a pointer of unknown type
{
	Mat img = (*((myData*)userdata)).variable1; //first we say that userdata is a pointer of Mat type, then we dereference to get the value of the actual type
	Mat b2;
	blur(img, b2, Size(trackpos, trackpos));
	imshow("Blur Window", b2);
	(*(myData*)userdata).variable2 = b2;
}

int test_trackbar2(int something) 
{
	Mat src1 = imread("..\\data_store\\david_1.jpg");
	Mat src2 = imread("..\\data_store\\david_2.jpg");
	resize(src2, src2, src1.size());
	if (!src1.data) { printf("Error loading src1 \n"); return -1; }
	if (!src2.data) { printf("Error loading src2 \n"); return -1; }
	myData holder;
	holder.variable1 = src1;
	holder.variable2 = src2;
	int b = 3; // blur value
	namedWindow("Blur Window");
	createTrackbar("blur", "Blur Window", &b, 100, onChange, (void*)(&holder)); 
	waitKey(0);

	namedWindow("Updated Image");
	imshow("Updated Image", holder.variable2);
	waitKey(0);
}
