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

struct trackbarDataExample {
	Mat src;
	Mat dst;
	int brightness;
	int contrast;
};

static void onChangeBrightness(int brightness, void *userdata) //void* mean that it is a pointer of unknown type
{
	Mat img = (*((trackbarDataExample*)userdata)).src; //first we say that userdata is a pointer of Mat type, then we dereference to get the value of the actual type
	Mat dst;
	(*((trackbarDataExample*)userdata)).brightness = brightness;
	//(*((trackbarDataExample*)userdata)).constrast = contrast;
	img.convertTo(dst, -1, (*((trackbarDataExample*)userdata)).contrast, (*((trackbarDataExample*)userdata)).brightness);
	imshow("Adjust Window", dst);
	(*(trackbarDataExample*)userdata).dst = dst;
}

static void onChangeContrast(int contrast, void *userdata) //void* mean that it is a pointer of unknown type
{
	Mat img = (*((trackbarDataExample*)userdata)).src; //first we say that userdata is a pointer of Mat type, then we dereference to get the value of the actual type
	Mat dst;
	//(*((trackbarDataExample*)userdata)).brightness = brightness;
	(*((trackbarDataExample*)userdata)).contrast = contrast;
	img.convertTo(dst, -1, (*((trackbarDataExample*)userdata)).contrast, (*((trackbarDataExample*)userdata)).brightness);
	imshow("Adjust Window", dst);
	(*(trackbarDataExample*)userdata).dst = dst;
}

int test_trackbar2(int something)
{
	Mat src = imread("..\\data_store\\david_1.jpg");  
	if (!src.data) { printf("Error loading src \n"); return -1; }
	Mat dst;

	trackbarDataExample holder;
	holder.src = src;
	holder.dst = dst;
	holder.brightness = 1;
	holder.contrast = 0;

	int brightness = 1;
	int contrast = 0;

	namedWindow("Adjust Window");
	cvCreateTrackbar2("Brightness", "Adjust Window", &brightness, 100, onChangeBrightness, (void*)(&holder));
	cvCreateTrackbar2("Contrast", "Adjust Window", &contrast, 100, onChangeContrast, (void*)(&holder));
	waitKey(0);

	namedWindow("Updated Image");
	imshow("Updated Image", holder.dst);
	waitKey(0);
}
