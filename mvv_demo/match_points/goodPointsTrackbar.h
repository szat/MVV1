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


struct dataTrackbarCorners {
	Mat image;
	vector<Point2f> corners;
	double qualityLevel = 0.01; 
	double minDistance = 10;
	int maxCorners = 1000;
	int blockSize = 3;
	double k = 0.04;
	bool useHarrisDetector = false;
};

static void onChangeTrackbarCorners(int slider, void *userdata) //void* mean that it is a pointer of unknown type
{
	Mat temp = (*((dataTrackbarCorners*)userdata)).image;
	goodFeaturesToTrack(
		(*((dataTrackbarCorners*)userdata)).image,
		(*((dataTrackbarCorners*)userdata)).corners,
		(*((dataTrackbarCorners*)userdata)).maxCorners,
		(*((dataTrackbarCorners*)userdata)).qualityLevel,
		(*((dataTrackbarCorners*)userdata)).minDistance = (double)slider,
		Mat(),
		(*((dataTrackbarCorners*)userdata)).blockSize,
		(*((dataTrackbarCorners*)userdata)).useHarrisDetector,
		(*((dataTrackbarCorners*)userdata)).k
	);

	int r = 4;
	Mat copy = (*((dataTrackbarCorners*)userdata)).image.clone();
	RNG rng(12345); //random number generator
	for (size_t i = 0; i < (*((dataTrackbarCorners*)userdata)).corners.size(); i++) {
		circle(copy, (*((dataTrackbarCorners*)userdata)).corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	imshow("Corners", copy);
	cout << "number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
}

int trackbarCorners(vector<Point2f>& corners)
{
	Mat src1 = imread("..\\data_store\\david_1.jpg", IMREAD_GRAYSCALE);
	if (!src1.data) { printf("Error loading src1 \n"); return -1; }

	
	double qualityLevel = 0.01;
	double minDistance = 10;
	int maxCorners = 1000;
	int blockSize = 3;
	double k = 0.04;
	bool useHarrisDetector = false;

	dataTrackbarCorners holder;
	holder.image = src1;
	holder.corners = corners;
	holder.qualityLevel = qualityLevel;
	holder.minDistance = minDistance;
	holder.maxCorners = maxCorners;
	holder.blockSize = blockSize;
	holder.k = k;
	holder.useHarrisDetector = useHarrisDetector;

	int passMinDistance = 1;
	namedWindow("Corners");
	createTrackbar("minDistance", "Corners", &passMinDistance, 100, onChangeTrackbarCorners, (void*)(&holder)); 
	cout << "Outside of trackbar, number of corners is: " << holder.corners.size() << endl;
	waitKey(0);

	//sending the information out of trackbar
	corners = holder.corners;
}

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


