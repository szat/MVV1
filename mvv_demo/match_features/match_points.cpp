/*
By downloading, copying, installing or using the software you agree to this license. If you do not agree to this license, do not download, install, copy or use the software.

License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
Neither the names of the copyright holders nor the names of the contributors may be used to endorse or promote products derived from this software without specific prior written permission.
This software is provided by the copyright holders and contributors “as is” and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall copyright holders or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
*/

// akaze_opencv.cpp : Defines the entry point for the console application.

#include "match_points.h"

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

const float inlier_threshold = 20.0f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
const double akaze_thresh = 3e-4;    // AKAZE detection threshold set to locate about 1000 keypoints

bool has_suffix(const std::string &str, const std::string &suffix)
{
	return (str.size() >= suffix.size() &&
		str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
}

bool has_image_suffix(const std::string &str) {
	return (has_suffix(str, ".jpg") || has_suffix(str, ".jpeg") || has_suffix(str, ".png") || has_suffix(str, ".bmp") || has_suffix(str, ".svg") || has_suffix(str, ".tiff") || has_suffix(str, ".ppm"));
}

void akaze_wrapper(float akaze_thresh, const Mat& img_in, vector<KeyPoint>& kpts_out, Mat& desc_out, bool verbose = false) {
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	time_t tstart, tend;
	tstart = time(0);
	akaze->detectAndCompute(img_in, noArray(), kpts_out, desc_out);
	tend = time(0);
	cout << "akaze_wrapper(thr=" << akaze_thresh << ",[h=" << img_in.size().height << ",w=" << img_in.size().width << "]) finished in " << difftime(tend, tstart) << "s and found " << kpts_out.size() << " features." << endl;
}

void ratio_matcher_wrapper(const float ratio, const vector<KeyPoint>& kpts1_in, const vector<KeyPoint>& kpts2_in, const Mat& desc1_in, const Mat& desc2_in, vector<KeyPoint>& kpts1_out, vector<KeyPoint>& kpts2_out, bool verbose = false) {
	time_t tstart, tend;
	vector<vector<DMatch>> matchesLoweRatio;
	BFMatcher matcher(NORM_HAMMING);
	tstart = time(0);
	matcher.knnMatch(desc1_in, desc2_in, matchesLoweRatio, 2);
	int nbMatches = matchesLoweRatio.size();
	for (int i = 0; i < nbMatches; i++) {
		DMatch first = matchesLoweRatio[i][0];
		float dist1 = matchesLoweRatio[i][0].distance;
		float dist2 = matchesLoweRatio[i][1].distance;
		if (dist1 < ratio * dist2) {
			kpts1_out.push_back(kpts1_in[first.queryIdx]);
			kpts2_out.push_back(kpts2_in[first.trainIdx]);
		}
	}
	tend = time(0);
	cout << "Ratio matching with BF(NORM_HAMMING) and ratio " << ratio << " finished in " << difftime(tend, tstart) << "s and matched " << kpts1_out.size() << " features." << endl;
}

void ransac_wrapper(const float ball_radius, const float inlier_thresh, const vector<KeyPoint>& kpts1_in, const vector<KeyPoint>& kpts2_in, Mat& homography_out, vector<KeyPoint>& kpts1_out, vector<KeyPoint>& kpts2_out, bool verbose = false) {
	cout << "RANSAC to estimate global homography with max deviating distance being " << ball_radius << "." << endl;

	vector<Point2f> keysImage1;
	vector<Point2f> keysImage2;
	vector<DMatch> good_matches;

	int nbMatches = kpts1_in.size();
	for (int i = 0; i < nbMatches; i++) {
		keysImage1.push_back(kpts1_in.at(i).pt);
		keysImage2.push_back(kpts2_in.at(i).pt);
	}

	Mat H = findHomography(keysImage1, keysImage2, CV_RANSAC, ball_radius);
	cout << "RANSAC found the homography." << endl;

	nbMatches = kpts1_in.size();
	for (int i = 0; i < nbMatches; i++) {
		Mat col = Mat::ones(3, 1, CV_64F);// , CV_32F);
		col.at<double>(0) = kpts1_in[i].pt.x;
		col.at<double>(1) = kpts1_in[i].pt.y;

		col = H * col;
		col /= col.at<double>(2); //because you are in projective space
		double dist = sqrt(pow(col.at<double>(0) - kpts2_in[i].pt.x, 2) + pow(col.at<double>(1) - kpts2_in[i].pt.y, 2));

		if (dist < inlier_thresh) {
			int new_i = static_cast<int>(kpts1_out.size());
			kpts1_out.push_back(kpts1_in[i]);
			kpts2_out.push_back(kpts2_in[i]);
		}
	}

	cout << "Homography filtering with inlier threshhold of " << inlier_thresh << " has matched " << kpts1_out.size() << " features." << endl;
}

int test_match_points(void)
{
	const float akaze_thr = 3e-4;    // AKAZE detection threshold set to locate about 1000 keypoints
	const float ratio = 0.8f;   // Nearest neighbor matching ratio
	const float inlier_thr = 20.0f; // Distance threshold to identify inliers
	const float ball_radius = 5;

	time_t ststart, tend;
	cout << "Welcome to match_points testing unit!" << endl;
	string address = "..\\data_store\\";
	string input = "";
	ifstream infile1;
	while (true) {
		cout << "Please enter the name of the first image in folder data_store:";
		getline(cin, input);
		address += input;
		cout << "The address you entered is " << address << endl;
		if (has_image_suffix(address)) {
			infile1.open(address.c_str());
			if (infile1) break;
			cout << "Invalid file." << endl;
			address = "..\\data_store\\";
			input = "";
		}
	}
	Mat img1 = imread(address, IMREAD_GRAYSCALE);

	address = "..\\data_store\\";
	input = "";
	ifstream infile2;
	while (true) {
		cout << "Please enter the name of the second image in folder data_store:";
		getline(cin, input);
		address += input;
		cout << "The address you entered is " << address << endl;
		if (has_image_suffix(address)) {
			infile2.open(address.c_str());
			if (infile2) break;
			cout << "Invalid file." << endl;
			address = "..\\data_store\\";
			input = "";
		}
	}
	Mat img2 = imread(address, IMREAD_GRAYSCALE);


	const char* source_window = "Image";
	Mat src, src_gray;
	src_gray = img1;

	vector<Point2f> corners;
	double qualityLevel = 0.01; //the higher, the less points
	double minDistance = 10;
	int maxCorners = 1000;
	RNG rng(12345);
	int blockSize = 3; //not sure of this effect
	double k = 0.04; //not sure of this effect
	bool useHarrisDetector = false;
	Mat copy;
	copy = img1.clone();
	goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	cout << "Number of good corners detected: " << corners.size() << "." << endl;
	int r = 4;
	for (size_t i = 0; i < corners.size(); i++) circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	namedWindow(source_window, WINDOW_AUTOSIZE);
	imshow(source_window, copy);
	waitKey(0);

	///////////////////////////////////////////////////////

	vector<KeyPoint> kpts1_step1;
	vector<KeyPoint> kpts2_step1;
	Mat desc1_step1;
	Mat desc2_step1;
	akaze_wrapper(akaze_thresh, img1, kpts1_step1, desc1_step1);
	akaze_wrapper(akaze_thresh, img2, kpts2_step1, desc2_step1);

	vector<KeyPoint> kpts1_step2;
	vector<KeyPoint> kpts2_step2;
	ratio_matcher_wrapper(ratio, kpts1_step1, kpts2_step1, desc1_step1, desc2_step1, kpts1_step2, kpts2_step2);

	Mat homography;
	vector<KeyPoint> kpts1_step3;
	vector<KeyPoint> kpts2_step3;
	ransac_wrapper(ball_radius, inlier_thr, kpts1_step2, kpts2_step2, homography, kpts1_step3, kpts2_step3);

	vector<DMatch> good_matches2;
	Mat res;
	for (int i = 0; i < kpts1_step3.size(); i++) good_matches2.push_back(DMatch(i, i, 0));
	drawMatches(img1, kpts1_step3, img2, kpts2_step3, good_matches2, res);
	imwrite("res.png", res);

	cout << endl << "A-KAZE Matching Results" << endl;
	cout << "*******************************" << endl;
	cout << "# Keypoints 1:                        \t" << kpts1_step1.size() << endl;
	cout << "# Keypoints 2:                        \t" << kpts2_step1.size() << endl;
	cout << "# Matches 1:                          \t" << kpts1_step2.size() << endl;
	cout << "# Matches 2:                          \t" << kpts2_step2.size() << endl;
	cout << "# Inliers 1:                          \t" << kpts1_step3.size() << endl;
	cout << "# Inliers 2:                          \t" << kpts2_step3.size() << endl;
	cout << "# Inliers Ratio:                      \t" << (float)kpts1_step3.size() / (float)kpts1_step2.size() << endl;
	cout << endl;

	cin.ignore();
	return 0;
}
