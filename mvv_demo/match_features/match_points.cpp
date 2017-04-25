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
#include <opencv2/ml/ml.hpp> //for knn
#include <vector>
#include <iostream>
#include <ctime>
#include <algorithm>

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

void akaze_script(float akaze_thresh, const Mat& img_in, vector<KeyPoint>& kpts_out, Mat& desc_out) {
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	time_t tstart, tend;
	tstart = time(0);
	akaze->detectAndCompute(img_in, noArray(), kpts_out, desc_out);
	tend = time(0);
	cout << "akaze_wrapper(thr=" << akaze_thresh << ",[h=" << img_in.size().height << ",w=" << img_in.size().width << "]) finished in " << difftime(tend, tstart) << "s and found " << kpts_out.size() << " features." << endl;
}


void ratio_matcher_script(const float ratio, const vector<KeyPoint>& kpts1_in, const vector<KeyPoint>& kpts2_in, const Mat& desc1_in, const Mat& desc2_in, vector<KeyPoint>& kpts1_out, vector<KeyPoint>& kpts2_out) {
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

void ransac_script(const float ball_radius, const float inlier_thresh, const vector<KeyPoint>& kpts1_in, const vector<KeyPoint>& kpts2_in, Mat& homography_out, vector<KeyPoint>& kpts1_out, vector<KeyPoint>& kpts2_out) {
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

vector<vector<KeyPoint>> match_points_mat(Mat img1, Mat img2)
{
	const float akaze_thr = 3e-4;    // AKAZE detection threshold set to locate about 1000 keypoints
	const float ratio = 0.8f;   // Nearest neighbor matching ratio
	const float inlier_thr = 20.0f; // Distance threshold to identify inliers
	const float ball_radius = 5;

	vector<KeyPoint> kpts1_step1;
	vector<KeyPoint> kpts2_step1;
	Mat desc1_step1;
	Mat desc2_step1;
	akaze_script(akaze_thresh, img1, kpts1_step1, desc1_step1);
	akaze_script(akaze_thresh, img2, kpts2_step1, desc2_step1);

	vector<KeyPoint> kpts1_step2;
	vector<KeyPoint> kpts2_step2;
	ratio_matcher_script(ratio, kpts1_step1, kpts2_step1, desc1_step1, desc2_step1, kpts1_step2, kpts2_step2);

	Mat homography;
	vector<KeyPoint> kpts1_step3;
	vector<KeyPoint> kpts2_step3;
	ransac_script(ball_radius, inlier_thr, kpts1_step2, kpts2_step2, homography, kpts1_step3, kpts2_step3);

	vector<vector<KeyPoint>> pointMatches = { kpts1_step3, kpts2_step3 };
	return pointMatches;
}

vector<vector<KeyPoint>> test_match_points_2(string imagePathA, string imagePathB)
{
	cout << "Welcome to match_points_2 testing unit!" << endl;
	string address = "..\\data_store\\" + imagePathA;
	string input = "";
	ifstream infile1;
	infile1.open(address.c_str());

	Mat img1 = imread(address, IMREAD_GRAYSCALE);
	Mat img1Display = imread(address);

	address = "..\\data_store\\" + imagePathB;
	input = "";
	ifstream infile2;
	infile2.open(address.c_str());

	Mat img2 = imread(address, IMREAD_GRAYSCALE);
	Mat img2Display = imread(address);

	//http://docs.opencv.org/trunk/da/d9b/group__features2d.html#ga15e1361bda978d83a2bea629b32dfd3c

	//find the other detectors on page http://docs.opencv.org/trunk/d5/d51/group__features2d__main.html
	Ptr<GFTTDetector> detectorGFTT = GFTTDetector::create();
	detectorGFTT->setMaxFeatures(100);
	detectorGFTT->setQualityLevel(0.1);
	detectorGFTT->setMinDistance(10);
	detectorGFTT->setBlockSize(5);
	detectorGFTT->setHarrisDetector(true);
	detectorGFTT->setK(0.2);

	//Find features
	vector<KeyPoint> kpts1;
	detectorGFTT->detect(img1, kpts1);
	cout << "GFTT found " << kpts1.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1;
	extractorORB->compute(img1, kpts1, desc1);

	//Find features
	vector<KeyPoint> kpts2;
	detectorGFTT->detect(img2, kpts2);
	cout << "GFTT found " << kpts2.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2;
	extractorORB->compute(img2, kpts2, desc2);
	
	//Ratio Matching
	float ratio = 0.8f;
	vector<KeyPoint> kptsRatio1;
	vector<KeyPoint> kptsRatio2;

	time_t tstart, tend;
	vector<vector<DMatch>> matchesLoweRatio;
	BFMatcher matcher(NORM_HAMMING);
	tstart = time(0);
	matcher.knnMatch(desc1, desc2, matchesLoweRatio, 2);
	int nbMatches = matchesLoweRatio.size();
	for (int i = 0; i < nbMatches; i++) {
		DMatch first = matchesLoweRatio[i][0];
		float dist1 = matchesLoweRatio[i][0].distance;
		float dist2 = matchesLoweRatio[i][1].distance;
		if (dist1 < ratio * dist2) {
			kptsRatio1.push_back(kpts1[first.queryIdx]);
			kptsRatio2.push_back(kpts2[first.trainIdx]);
		}
	}
	tend = time(0);
	cout << "Ratio matching with BF(NORM_HAMMING) and ratio " << ratio << " finished in " << difftime(tend, tstart) << "s and matched " << kptsRatio1.size() << " features." << endl;

	//-- Draw matches
	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kptsRatio1.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsRatio1, img2Display, kptsRatio2, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);

	//-- Knn for balling

	vector<Point2f> ptsKnn2;
	vector<Point2f> ptsKnn1;

	int nbPtsKnn1 = kptsRatio1.size();
	for (int i = 0; i < nbPtsKnn1; i++) ptsKnn1.push_back(kptsRatio1.at(i).pt);

	int nbPtsKnn2 = kptsRatio2.size();
	for (int i = 0; i < nbPtsKnn2; i++) ptsKnn2.push_back(kptsRatio2.at(i).pt);

	cv::flann::KDTreeIndexParams indexParams1;
	cv::flann::Index kdtree(cv::Mat(ptsKnn1).reshape(1), indexParams1);
	vector<int> indicesKnn1;
	vector<float> distsKnn1;
	vector<float> query; query.push_back(ptsKnn1.at(0).x); query.push_back(ptsKnn2.at(0).y); 
	kdtree.radiusSearch(query, indicesKnn1, distsKnn1, 30, 100, cv::flann::SearchParams(64));

	cout << "Knn found " << indicesKnn1.size() << " and " << distsKnn1.size() << " keypoints in a " << 30 << " radius ball. " << endl;

	cout << "Little Development Testing" << endl;
	vector<int> zero = { 0 };
	vector<int> myvector = { 0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0 };

	while (!myvector.empty() && (myvector.back() == 0)) {
		myvector.pop_back();
	}

	for (auto const& value : myvector) cout << value << endl;

	//-- RANSAC homography estimation and keypoints filtering
	float ballRadius = 5;
	float inlierThresh = 50;
	cout << "RANSAC to estimate global homography with max deviating distance being " << ballRadius << "." << endl;

	vector<Point2f> ptsRansac1;
	vector<Point2f> ptsRansac2;
	vector<DMatch> good_matches;
	vector<KeyPoint> kptsRansac1;
	vector<KeyPoint> kptsRansac2;

	nbMatches = kptsRatio1.size();
	for (int i = 0; i < nbMatches; i++) {
		ptsRansac1.push_back(kptsRatio1.at(i).pt);
		ptsRansac2.push_back(kptsRatio2.at(i).pt);
	}

	Mat H = findHomography(ptsRansac1, ptsRansac2, CV_RANSAC, ballRadius);
	cout << "RANSAC found the homography." << endl;

	nbMatches = ptsRansac1.size();
	for (int i = 0; i < nbMatches; i++) {
		Mat col = Mat::ones(3, 1, CV_64F);// , CV_32F);
		col.at<double>(0) = kptsRatio1[i].pt.x;
		col.at<double>(1) = kptsRatio2[i].pt.y;

		col = H * col;
		col /= col.at<double>(2); //because you are in projective space
		double dist = sqrt(pow(col.at<double>(0) - kptsRatio2[i].pt.x, 2) + pow(col.at<double>(1) - kptsRatio2[i].pt.y, 2));

		if (dist < inlierThresh) {
			int new_i = static_cast<int>(kptsRansac1.size());
			kptsRansac1.push_back(kptsRatio1[i]);
			kptsRansac2.push_back(kptsRatio2[i]);
		}
	}

	cout << "Homography filtering with inlier threshhold of " << inlierThresh << " has matched " << kptsRansac1.size() << " features." << endl;

	//-- Draw matches
	Mat img1to2Ransac;
	vector<DMatch> matchesIndexTrivialRansac;
	for (int i = 0; i < kptsRansac1.size(); i++) matchesIndexTrivialRansac.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsRansac1, img2Display, kptsRansac2, matchesIndexTrivialRansac, img1to2Ransac);

	//-- Show detected matches
	imshow("Matches", img1to2Ransac);

	waitKey(0);



	return{ kptsRansac1, kptsRansac2 };
}

vector<vector<KeyPoint>> test_match_points(string imagePathA, string imagePathB)
{
	const float akaze_thr = 3e-4;    // AKAZE detection threshold set to locate about 1000 keypoints
	const float ratio = 0.8f;   // Nearest neighbor matching ratio
	const float inlier_thr = 20.0f; // Distance threshold to identify inliers
	const float ball_radius = 5;

	time_t ststart, tend;
	cout << "Welcome to match_points testing unit!" << endl;
	string address = "..\\data_store\\" + imagePathA;
	string input = "";
	ifstream infile1;
	infile1.open(address.c_str());

	Mat img1 = imread(address, IMREAD_GRAYSCALE);

	address = "..\\data_store\\" + imagePathB;
	input = "";
	ifstream infile2;
	infile2.open(address.c_str());

	Mat img2 = imread(address, IMREAD_GRAYSCALE);

	vector<KeyPoint> kpts1_step1;
	vector<KeyPoint> kpts2_step1;
	Mat desc1_step1;
	Mat desc2_step1;
	akaze_script(akaze_thresh, img1, kpts1_step1, desc1_step1);
	akaze_script(akaze_thresh, img2, kpts2_step1, desc2_step1);

	vector<KeyPoint> kpts1_step2;
	vector<KeyPoint> kpts2_step2;
	ratio_matcher_script(ratio, kpts1_step1, kpts2_step1, desc1_step1, desc2_step1, kpts1_step2, kpts2_step2);

	Mat homography;
	vector<KeyPoint> kpts1_step3;
	vector<KeyPoint> kpts2_step3;
	ransac_script(ball_radius, inlier_thr, kpts1_step2, kpts2_step2, homography, kpts1_step3, kpts2_step3);

	vector<vector<KeyPoint>> pointMatches = { kpts1_step3, kpts2_step3 };
	return pointMatches;
}

