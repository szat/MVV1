#include "good_features.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
	int maxCorners = 1000;
	double qualityLevel = 0.01;
	int MAXQUALITYLEVEL = 1000;
	double minDistance = 10;
	int MAXMINDISTANCE = 10;
	int blockSize = 3;
	double k = 0.04;
	int MAXK = 1000;
	bool useHarrisDetector = false;
};

void changeCornersMaxCorners(int maxCornersSlider, void *userdata) {
	goodFeaturesToTrack(
		(*((dataTrackbarCorners*)userdata)).image,
		(*((dataTrackbarCorners*)userdata)).corners,
		(*((dataTrackbarCorners*)userdata)).maxCorners = maxCornersSlider,
		(*((dataTrackbarCorners*)userdata)).qualityLevel,
		(*((dataTrackbarCorners*)userdata)).minDistance,
		Mat(),
		(*((dataTrackbarCorners*)userdata)).blockSize,
		(*((dataTrackbarCorners*)userdata)).useHarrisDetector,
		(*((dataTrackbarCorners*)userdata)).k
	);
	int r = 3;

	Mat copy_rbg((*((dataTrackbarCorners*)userdata)).image.size(), CV_8UC3);
	cvtColor((*((dataTrackbarCorners*)userdata)).image, copy_rbg, CV_GRAY2RGB);
	RNG rng(12345); //random number generator
	for (size_t i = 0; i < (*((dataTrackbarCorners*)userdata)).corners.size(); i++) {
		circle(copy_rbg, (*((dataTrackbarCorners*)userdata)).corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	imshow("Display", copy_rbg);
	cout << "number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
}
void changeCornersQualityLevel(int qualityLevelInt, void *userdata) {
	int MAXQUALITYLEVEL = (*((dataTrackbarCorners*)userdata)).MAXQUALITYLEVEL;
	goodFeaturesToTrack(
		(*((dataTrackbarCorners*)userdata)).image,
		(*((dataTrackbarCorners*)userdata)).corners,
		(*((dataTrackbarCorners*)userdata)).maxCorners,
		(*((dataTrackbarCorners*)userdata)).qualityLevel = 0.000000001 + ((double) qualityLevelInt) / MAXQUALITYLEVEL,
		(*((dataTrackbarCorners*)userdata)).minDistance,
		Mat(),
		(*((dataTrackbarCorners*)userdata)).blockSize,
		(*((dataTrackbarCorners*)userdata)).useHarrisDetector,
		(*((dataTrackbarCorners*)userdata)).k
	);
	int r = 3;
	Mat copy_rbg((*((dataTrackbarCorners*)userdata)).image.size(), CV_8UC3);
	cvtColor((*((dataTrackbarCorners*)userdata)).image, copy_rbg, CV_GRAY2RGB);
	RNG rng(12345); //random number generator
	for (size_t i = 0; i < (*((dataTrackbarCorners*)userdata)).corners.size(); i++) {
		circle(copy_rbg, (*((dataTrackbarCorners*)userdata)).corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	imshow("Display", copy_rbg);
	cout << "number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
}
void changeCornersMinDistance(int minDistanceInt, void *userdata) {
	int MAXMINDISTANCE = (*((dataTrackbarCorners*)userdata)).MAXMINDISTANCE;
	goodFeaturesToTrack(
		(*((dataTrackbarCorners*)userdata)).image,
		(*((dataTrackbarCorners*)userdata)).corners,
		(*((dataTrackbarCorners*)userdata)).maxCorners,
		(*((dataTrackbarCorners*)userdata)).qualityLevel,
		(*((dataTrackbarCorners*)userdata)).minDistance = 0.000000001 + (double) ((double) minDistanceInt / MAXMINDISTANCE),
		Mat(),
		(*((dataTrackbarCorners*)userdata)).blockSize,
		(*((dataTrackbarCorners*)userdata)).useHarrisDetector,
		(*((dataTrackbarCorners*)userdata)).k
	);
	int r = 3;
	//Mat copy = (*((dataTrackbarCorners*)userdata)).image.clone();
	
	Mat copy_rbg((*((dataTrackbarCorners*)userdata)).image.size(), CV_8UC3);
	cvtColor((*((dataTrackbarCorners*)userdata)).image, copy_rbg, CV_GRAY2RGB);
	RNG rng(12345); //random number generator
	for (size_t i = 0; i < (*((dataTrackbarCorners*)userdata)).corners.size(); i++) {
		circle(copy_rbg, (*((dataTrackbarCorners*)userdata)).corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	imshow("Display", copy_rbg);
	cout << "number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
}
void changeCornersBlockSize(int blockSizeSlider, void *userdata) {
		goodFeaturesToTrack(
			(*((dataTrackbarCorners*)userdata)).image,
			(*((dataTrackbarCorners*)userdata)).corners,
			(*((dataTrackbarCorners*)userdata)).maxCorners,
			(*((dataTrackbarCorners*)userdata)).qualityLevel,
			(*((dataTrackbarCorners*)userdata)).minDistance,
			Mat(),
			(*((dataTrackbarCorners*)userdata)).blockSize = blockSizeSlider,
			(*((dataTrackbarCorners*)userdata)).useHarrisDetector,
			(*((dataTrackbarCorners*)userdata)).k
		);
		int r = 3;

		Mat copy_rbg((*((dataTrackbarCorners*)userdata)).image.size(), CV_8UC3);
		cvtColor((*((dataTrackbarCorners*)userdata)).image, copy_rbg, CV_GRAY2RGB);
		RNG rng(12345); //random number generator
		for (size_t i = 0; i < (*((dataTrackbarCorners*)userdata)).corners.size(); i++) {
			circle(copy_rbg, (*((dataTrackbarCorners*)userdata)).corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
		}
		imshow("Display", copy_rbg);
		cout << "number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
	
}
void changeCornersKInt(int kInt, void *userdata) {}



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
	
	namedWindow("Controls");
	namedWindow("Display");

	int passQualityLevel = 1;
	cvCreateTrackbar2("qualityLevel(Scaled)", "Controls", &passQualityLevel, holder.MAXQUALITYLEVEL, changeCornersQualityLevel, (void*)(&holder));

	int passMinDistance = 1;
	cvCreateTrackbar2("minDistance(Scaled)", "Controls", &passMinDistance, 1000, changeCornersMinDistance, (void*)(&holder));

	int passMaxCorners = 1;
	cvCreateTrackbar2("maxCorners", "Controls", &passMaxCorners, 1000, changeCornersMaxCorners, (void*)(&holder));

	int passBlockSize = 1;
	cvCreateTrackbar2("blockSize", "Controls", &passBlockSize, 10, changeCornersBlockSize, (void*)(&holder));

	cout << "Outside of trackbar, number of corners is: " << holder.corners.size() << endl;
	waitKey(0);

	//sending the information out of trackbar
	corners = holder.corners;
}
