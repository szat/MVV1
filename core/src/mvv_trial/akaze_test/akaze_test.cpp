// mvv_trial.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main(void)
{
	string readpath = "..\\data_store\\lena.bmp";
	int key;
	Mat cam_frame_col, cam_frame_gray;          //Camera frames in color and grayscale (e.g 'template')
	Mat descriptors_cam;                        //Descriptors for camera frame (e.g. 'template')
	std::vector<KeyPoint> kp_cam;               //Keypoints for camera frame (e.g. 'template')


												//-- AKAZE parameters
	float inlier_threshold = 2.5;
	float match_ratio = 0.8;

	key = 0;

	BFMatcher matcher(NORM_HAMMING);

	Ptr<AKAZE> detectorAKAZE = AKAZE::create();

	//-- Importing image
	cam_frame_col = imread(readpath);

	//-- Processing the camera frame to grayscale.
	cvtColor(cam_frame_col, cam_frame_gray, CV_BGR2GRAY);

	imshow("Gray", cam_frame_gray);
	waitKey(30);

	//-- Find keypoints for camera frame.
	detectorAKAZE->detectAndCompute(cam_frame_gray, noArray(), kp_cam, descriptors_cam);

	return 0;
}