#include "good_features.h"

using namespace std;
using namespace cv;

struct dataTrackbarCorners {
	Mat image;
	vector<Point2f> corners;

	int maxCorners = 1000;
	int blockSize = 3;
	double qualityLevel = 0.001;
	double minDistance = 0.01;
	double k = 0.01;

	double qualityPrecision = 0.001;
	double minDistancePrecision = 0.01;
	double kPrecision = 0.00001;

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
	cout << "With maxCorners = " << maxCornersSlider << ", number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
}

void changeCornersBlockSize(int blockSizeSlider, void *userdata) {
	goodFeaturesToTrack(
		(*((dataTrackbarCorners*)userdata)).image,
		(*((dataTrackbarCorners*)userdata)).corners,
		(*((dataTrackbarCorners*)userdata)).maxCorners,
		(*((dataTrackbarCorners*)userdata)).qualityLevel,
		(*((dataTrackbarCorners*)userdata)).minDistance,
		Mat(),
		//We don't want to have blockSize == 0, however I don't see how I can tell the trackbar not to touch 0
		(blockSizeSlider > 0) ? (*((dataTrackbarCorners*)userdata)).blockSize = blockSizeSlider : (*((dataTrackbarCorners*)userdata)).blockSize,
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
	cout << "With blockSize = " << blockSizeSlider << ", number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;

}

void changeCornersQualityLevel(int qualityLevelInt, void *userdata) {
	
	goodFeaturesToTrack(
		(*((dataTrackbarCorners*)userdata)).image,
		(*((dataTrackbarCorners*)userdata)).corners,
		(*((dataTrackbarCorners*)userdata)).maxCorners,
		(*((dataTrackbarCorners*)userdata)).qualityLevel = 0.000000001 + (*((dataTrackbarCorners*)userdata)).qualityPrecision * qualityLevelInt,
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
	cout << "With qualityLevel = " << (*((dataTrackbarCorners*)userdata)).qualityLevel << ", number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
}

void changeCornersMinDistance(int minDistanceInt, void *userdata) {
	
	goodFeaturesToTrack(
		(*((dataTrackbarCorners*)userdata)).image,
		(*((dataTrackbarCorners*)userdata)).corners,
		(*((dataTrackbarCorners*)userdata)).maxCorners,
		(*((dataTrackbarCorners*)userdata)).qualityLevel,
		(*((dataTrackbarCorners*)userdata)).minDistance = 0.000000001 + (*((dataTrackbarCorners*)userdata)).minDistancePrecision * minDistanceInt,
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
	cout << "With minDistance = " << (*((dataTrackbarCorners*)userdata)).minDistance << ", number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
}

//Useless method unless useHarrisDetector == true
void changeCornersKInt(int kInt, void *userdata) {
	goodFeaturesToTrack(
		(*((dataTrackbarCorners*)userdata)).image,
		(*((dataTrackbarCorners*)userdata)).corners,
		(*((dataTrackbarCorners*)userdata)).maxCorners,
		(*((dataTrackbarCorners*)userdata)).qualityLevel,
		(*((dataTrackbarCorners*)userdata)).minDistance,
		Mat(),
		(*((dataTrackbarCorners*)userdata)).blockSize,
		(*((dataTrackbarCorners*)userdata)).useHarrisDetector,
		(*((dataTrackbarCorners*)userdata)).k = 0.000000001 + (*((dataTrackbarCorners*)userdata)).kPrecision * kInt
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
	cout << "With k = " << (*((dataTrackbarCorners*)userdata)).k << ", number of corners found is " << (*((dataTrackbarCorners*)userdata)).corners.size() << endl;
}

int trackbar_corners(string imagePath, vector<Point2f>& corners)
{
	Mat src1 = imread(imagePath, IMREAD_GRAYSCALE);
	if (!src1.data) { printf("Error loading src1 \n"); return -1; }

	double qualityLevel = 0.01;
	double minDistance = 10;
	int maxCorners = 1000;
	int blockSize = 3;
	double k = 0.04;
	bool useHarrisDetector = true;

	dataTrackbarCorners holder;
	holder.image = src1;
	holder.corners = corners;
	
	namedWindow("Controls");
	namedWindow("Display");

	//The quality level has to be between 0 and 1
	int passQualityLevel = 0;
	cvCreateTrackbar2("qual(S)", "Controls", &passQualityLevel, 300, changeCornersQualityLevel, (void*)(&holder));

	int passMinDistance = 0;
	cvCreateTrackbar2("minD(S)", "Controls", &passMinDistance, 5000, changeCornersMinDistance, (void*)(&holder));

	int passMaxCorners = 0;
	cvCreateTrackbar2("maxCrns", "Controls", &passMaxCorners, 5000, changeCornersMaxCorners, (void*)(&holder));

	int passBlockSize = 1;
	cvCreateTrackbar2("block", "Controls", &passBlockSize, 50, changeCornersBlockSize, (void*)(&holder));

	//K is the free parameter in the Harris detector, but we will use useHarrisDetector = false
	int passK = 0;
	cvCreateTrackbar2("k(S)", "Controls", &passK, 1000, changeCornersKInt, (void*)(&holder));

	cout << "Outside of trackbar, number of corners is: " << holder.corners.size() << endl;
	waitKey(0);

	//sending the information out of trackbar
	corners = holder.corners;
}
