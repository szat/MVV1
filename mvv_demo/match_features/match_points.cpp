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
#include <math.h>       /* fabs */
#include <iterator> //back_inserter
#include <string>

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

void ransac_script_against_hom(const float ball_radius, const float inlier_thresh, const vector<KeyPoint>& kpts1_in, const vector<KeyPoint>& kpts2_in, Mat& homography_in, vector<KeyPoint>& kpts1_out, vector<KeyPoint>& kpts2_out) {
	cout << "RANSAC to estimate global homography with max deviating distance being " << ball_radius << "." << endl;

	vector<Point2f> keysImage1;
	vector<Point2f> keysImage2;
	vector<DMatch> good_matches;

	int nbMatches = kpts1_in.size();
	for (int i = 0; i < nbMatches; i++) {
		keysImage1.push_back(kpts1_in.at(i).pt);
		keysImage2.push_back(kpts2_in.at(i).pt);
	}

	//Mat H = findHomography(keysImage1, keysImage2, CV_RANSAC, ball_radius);
	cout << "RANSAC found the homography." << endl;

	nbMatches = kpts1_in.size();
	for (int i = 0; i < nbMatches; i++) {
		Mat col = Mat::ones(3, 1, CV_64F);// , CV_32F);
		col.at<double>(0) = kpts1_in[i].pt.x;
		col.at<double>(1) = kpts1_in[i].pt.y;

		//col = H * col;
		col = homography_in * col;
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
	homography_out = H;

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

void ransac_filtering(float param, const vector<KeyPoint> & kptsDomain, const vector<KeyPoint> & kptsTarget, const vector<int> & indicesDomain, const vector<int> & indicesTarget, vector<int> indicesDomainPass, vector<int> indicesDomainFail, vector<int> indicesTargetPass, vector<int> indicesTargetFail) {
	vector<Point2f> ptsRansacDomain;
	vector<Point2f> ptsRansacTarget;
	//vector<DMatch> good_matches;

	vector<KeyPoint> kptsRansacNbh1;
	vector<KeyPoint> kptsRansacNbh2;

	int nbMatches = kptsDomain.size();
	for (int i = 0; i < nbMatches; i++) {
		ptsRansacDomain.push_back(kptsDomain.at(i).pt);
		ptsRansacTarget.push_back(kptsTarget.at(i).pt);
	}

	//http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findhomography#findhomography
	Mat H = findHomography(ptsRansacDomain, ptsRansacTarget, CV_RANSAC, param);
	
	for (int i = 0; i < nbMatches; i++) {
		Mat col = Mat::ones(3, 1, CV_64F);// , CV_32F);
		col.at<double>(0) = kptsDomain.at(i).pt.x;
		col.at<double>(1) = kptsDomain.at(i).pt.y;
		col = H * col;
		col /= col.at<double>(2); //because you are in projective space
		double dist = sqrt(pow(col.at<double>(0) - kptsTarget.at(i).pt.x, 2) + pow(col.at<double>(1) - kptsTarget.at(i).pt.y, 2));
		if (dist < param) { //I this it is correct in thinking that this is the same as the parameter in findHomography
			//int new_i = static_cast<int>(kptsRansacNbh1.size());
			//good_matches.push_back(DMatch(new_i, new_i, 0));

			indicesDomainPass.push_back(indicesDomain.at(i));
			indicesTargetPass.push_back(indicesTarget.at(i));
		}
		else {
			indicesDomainFail.push_back(indicesDomain.at(i));
			indicesTargetFail.push_back(indicesTarget.at(i));
		}
	}
}

void test_GFTT(string imagePathA, string imagePathB) {
	cout << "Welcome to match_GFTT testing unit!" << endl;
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

	//Set Detector
	//Find the other detectors on page http://docs.opencv.org/trunk/d5/d51/group__features2d__main.html

	Ptr<GFTTDetector> detectorGFTT = GFTTDetector::create();
	detectorGFTT->setMaxFeatures(5000);
	detectorGFTT->setQualityLevel(0.00000001);
	detectorGFTT->setMinDistance(0.1);
	detectorGFTT->setBlockSize(5);
	detectorGFTT->setHarrisDetector(false);
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

	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kptsRatio1.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsRatio1, img2Display, kptsRatio2, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);

}

void test_BRISK(string imagePathA, string imagePathB) {
	cout << "Welcome to match_BRISK testing unit!" << endl;
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

	//Set Detector
	//Find the other detectors on page http://docs.opencv.org/trunk/d5/d51/group__features2d__main.html

	Ptr<BRISK> detectorBRISK = BRISK::create();
	
	//Find features
	vector<KeyPoint> kpts1;
	detectorBRISK->detect(img1, kpts1);
	cout << "GFTT found " << kpts1.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1;
	detectorBRISK->compute(img1, kpts1, desc1); 
	//extractorORB->compute(img1, kpts1, desc1);

	//Find features
	vector<KeyPoint> kpts2;
	detectorBRISK->detect(img2, kpts2);
	cout << "GFTT found " << kpts2.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2;
	detectorBRISK->compute(img2, kpts2, desc2); 
	//extractorORB->compute(img2, kpts2, desc2);

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

	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kptsRatio1.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsRatio1, img2Display, kptsRatio2, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);
}

void affine_skew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai)
{
	cout << "Applying affine_skew(tilt = " << tilt << ", phi = " << phi << ") ..." << endl;

	int h = img.rows;
	int w = img.cols;

	mask = Mat(h, w, CV_8UC1, Scalar(255));

	Mat A = Mat::eye(2, 3, CV_32F);

	if (phi != 0.0)
	{
		phi *= M_PI / 180.;
		double s = sin(phi);
		double c = cos(phi);

		A = (Mat_<float>(2, 2) << c, -s, s, c);

		Mat corners = (Mat_<float>(4, 2) << 0, 0, w, 0, w, h, 0, h);
		Mat tcorners = corners*A.t();
		Mat tcorners_x, tcorners_y;
		tcorners.col(0).copyTo(tcorners_x);
		tcorners.col(1).copyTo(tcorners_y);
		std::vector<Mat> channels;
		channels.push_back(tcorners_x);
		channels.push_back(tcorners_y);
		merge(channels, tcorners);

		Rect rect = boundingRect(tcorners);
		A = (Mat_<float>(2, 3) << c, -s, -rect.x, s, c, -rect.y);

		warpAffine(img, img, A, Size(rect.width, rect.height), INTER_LINEAR, BORDER_REPLICATE);
	}
	if (tilt != 1.0)
	{
		double s = 0.8*sqrt(tilt*tilt - 1);
		GaussianBlur(img, img, Size(0, 0), s, 0.01);
		resize(img, img, Size(0, 0), 1.0 / tilt, 1.0, INTER_NEAREST);
		A.row(0) = A.row(0) / tilt;
	}
	if (tilt != 1.0 || phi != 0.0)
	{
		h = img.rows;
		w = img.cols;
		warpAffine(mask, mask, A, Size(w, h), INTER_NEAREST);
	}
	invertAffineTransform(A, Ai);
}

void affine_ORB_detect_and_compute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors)
{
	cout << "Applying affine_ORB_detect_and_compute(img, keypoints, descriptors)..." << endl;

	keypoints.clear();
	descriptors = Mat(0, 128, CV_32F);
	for (int tl = 1; tl < 6; tl++)
	{
		double t = pow(2, 0.5*tl);
		for (int phi = 0; phi < 180; phi += 72.0 / t)
		{
			std::vector<KeyPoint> kps;
			Mat desc;

			Mat timg, mask, Ai;
			img.copyTo(timg);

			affine_skew(t, phi, timg, mask, Ai);

#if 0
			Mat img_disp;
			bitwise_and(mask, timg, img_disp);
			namedWindow("Skew", WINDOW_AUTOSIZE);// Create a window for display.
			imshow("Skew", img_disp);
			waitKey(0);
#endif
			const double akaze_thresh = 3e-4;    // AKAZE detection threshold set to locate about 1000 keypoints

			Ptr<ORB> detectorORB = ORB::create(5000);

			detectorORB->detect(timg, kps, mask);
			detectorORB->compute(timg, kps, desc);
			cout << "detectorORB got " << kps.size() << " keypoints." << endl;
			/*
			Ptr<AKAZE> akaze = AKAZE::create();
			akaze->setThreshold(akaze_thresh);

			akaze->detect(timg, kps, mask);
			akaze->compute(timg, kps, desc);
			*/

			for (unsigned int i = 0; i < kps.size(); i++)
			{
				Point3f kpt(kps[i].pt.x, kps[i].pt.y, 1);
				Mat kpt_t = Ai*Mat(kpt);
				kps[i].pt.x = kpt_t.at<float>(0, 0);
				kps[i].pt.y = kpt_t.at<float>(1, 0);
			}
			keypoints.insert(keypoints.end(), kps.begin(), kps.end());
			descriptors.push_back(desc);
		}
	}
}

void test_affine_ORB(std::string imagePathA, std::string imagePathB) {
	cout << "Welcome to match_ORB testing unit!" << endl;
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

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;
	affine_ORB_detect_and_compute(img1, kpts1, desc1);
	affine_ORB_detect_and_compute(img2, kpts2, desc2);

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

	//-- RANSAC homography estimation and keypoints filtering
	float ballRadius = 10;
	float inlierThresh = 10;
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
			good_matches.push_back(DMatch(new_i, new_i, 0));
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
}

void test_ORB(string imagePathA, string imagePathB) {
	cout << "Welcome to match_ORB testing unit!" << endl;
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

	//Set Detector
	//Find the other detectors on page http://docs.opencv.org/trunk/d5/d51/group__features2d__main.html

	Ptr<ORB> detectorORB = ORB::create(10000);

	//Find features
	vector<KeyPoint> kpts1;
	detectorORB->detect(img1, kpts1);
	cout << "GFTT found " << kpts1.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1;
	detectorORB->compute(img1, kpts1, desc1);
	//extractorORB->compute(img1, kpts1, desc1);

	//Find features
	vector<KeyPoint> kpts2;
	detectorORB->detect(img2, kpts2);
	cout << "GFTT found " << kpts2.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2;
	detectorORB->compute(img2, kpts2, desc2);
	//extractorORB->compute(img2, kpts2, desc2);

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

	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kptsRatio1.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsRatio1, img2Display, kptsRatio2, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);
}

void test_FAST(string imagePathA, string imagePathB) {
	cout << "Welcome to match_FAST testing unit!" << endl;
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

	//Set Detector
	//Find the other detectors on page http://docs.opencv.org/trunk/d5/d51/group__features2d__main.html

	Ptr<FastFeatureDetector> detectorFAST = FastFeatureDetector::create();

	//Find features
	vector<KeyPoint> kpts1;
	detectorFAST->detect(img1, kpts1);
	cout << "GFTT found " << kpts1.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1;
	//detectorFAST->compute(img1, kpts1, desc1);, not implemented
	extractorORB->compute(img1, kpts1, desc1);

	//Find features
	vector<KeyPoint> kpts2;
	detectorFAST->detect(img2, kpts2);
	cout << "GFTT found " << kpts2.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2;
	//detectorFAST->compute(img2, kpts2, desc2);
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

	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kptsRatio1.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsRatio1, img2Display, kptsRatio2, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);
}

void test_AGAST(string imagePathA, string imagePathB) {
	cout << "Welcome to match_GFTT testing unit!" << endl;
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

	//Set Detector
	//Find the other detectors on page http://docs.opencv.org/trunk/d5/d51/group__features2d__main.html

	Ptr<AgastFeatureDetector> detectorAgast = AgastFeatureDetector::create();
	detectorAgast->setNonmaxSuppression(false); //both false and true give disappointing results

	//Find features
	vector<KeyPoint> kpts1;
	detectorAgast->detect(img1, kpts1);
	cout << "GFTT found " << kpts1.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1;
	//detectorAgast->compute(img1, kpts1, desc1); NOT IMPLEMENTED!
	extractorORB->compute(img1, kpts1, desc1);

	//Find features
	vector<KeyPoint> kpts2;
	detectorAgast->detect(img2, kpts2);
	cout << "GFTT found " << kpts2.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2;
	//detectorAgast->compute(img2, kpts2, desc2); NOT IMPLEMENTED!
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

	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kptsRatio1.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsRatio1, img2Display, kptsRatio2, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);
}

void council_filter(const std::vector<cv::KeyPoint> & kptsDomain, const std::vector<cv::KeyPoint> & kptsTarget, std::vector<cv::KeyPoint> & kptsDomainFiltered, std::vector<cv::KeyPoint> & kptsTargetFiltered, int numNeighbors, float minNumber) {
	if (kptsDomain.size() != kptsTarget.size()) {
		cout << "The two set of keypoints do not have the same length!" << endl;
		return;
	}
	int maxKpts = kptsDomain.size();

	vector<Point2f> ptsKnn1;
	vector<Point2f> ptsKnn2;

	for (int i = 0; i < kptsDomain.size(); i++) ptsKnn1.push_back(kptsDomain.at(i).pt);
	for (int i = 0; i < kptsTarget.size(); i++) ptsKnn2.push_back(kptsTarget.at(i).pt);

	cv::flann::KDTreeIndexParams indexParams1;
	cv::flann::Index kdtree1(cv::Mat(ptsKnn1).reshape(1), indexParams1);

	cv::flann::KDTreeIndexParams indexParams2;
	cv::flann::Index kdtree2(cv::Mat(ptsKnn2).reshape(1), indexParams2);

	float radius = 5;

	for (int i = 0; i < kptsDomain.size(); i++) {
		vector<int> indicesKnn1;
		vector<float> distsKnn1;
		vector<int> indicesKnn2;
		vector<float> distsKnn2;
		vector<float> query;

		query.push_back(kptsDomain.at(i).pt.x);
		query.push_back(kptsDomain.at(i).pt.y);
		kdtree1.knnSearch(query, indicesKnn1, distsKnn1, numNeighbors);
		kdtree2.knnSearch(query, indicesKnn2, distsKnn2, numNeighbors);
		sort(indicesKnn1.begin(), indicesKnn1.end());
		sort(indicesKnn2.begin(), indicesKnn2.end());

		vector<int> indexIntersection;
		set_intersection(indicesKnn1.begin(), indicesKnn1.end(), indicesKnn2.begin(), indicesKnn2.end(), back_inserter(indexIntersection));
		//cout << "index intersection of nbh is " << indexIntersection.size() << endl;
		if (indexIntersection.size() > minNumber) {
			//cout << "inside the if statement" << endl;
			kptsDomainFiltered.push_back(kptsDomain.at(i));
			kptsTargetFiltered.push_back(kptsTarget.at(i));
		}
	}
}

void ransac_filter(std::vector<cv::KeyPoint> kptsDomain, std::vector<cv::KeyPoint> kptsTarget, std::vector<cv::KeyPoint> kptsDomainFiltered, std::vector<cv::KeyPoint> kptsTargetFiltered, float inlierThreshhold) {

}

void test_akaze_harris_global_harris_local(std::string imagePathA, std::string imagePathB) {
	//First Apply Akaze
	//Second do a global matching with many GFTT points, i.e. ratio and then RANSAC using the AKAZE homography
	//Do a nbh growth

	//this code take the akaze matched keypoints in two images, and tries does one nbh growth (one time)
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
	cout << "kpts1_step3 size is " << kpts1_step3.size() << endl;
	cout << "kpts2_step3 size is " << kpts2_step3.size() << endl;

	Ptr<GFTTDetector> detectorGFTT = GFTTDetector::create();
	detectorGFTT->setMaxFeatures(20000);
	detectorGFTT->setQualityLevel(0.00001);
	detectorGFTT->setMinDistance(1);
	detectorGFTT->setBlockSize(3);
	detectorGFTT->setHarrisDetector(true);
	detectorGFTT->setK(0.2);

	//Find features
	vector<KeyPoint> kpts1_new;
	detectorGFTT->detect(img1, kpts1_new);
	cout << "GFTT found " << kpts1_new.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1_new;
	extractorORB->compute(img1, kpts1_new, desc1_new);

	//Find features
	vector<KeyPoint> kpts2_new;
	detectorGFTT->detect(img2, kpts2_new);
	cout << "GFTT found " << kpts2_new.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2_new;
	extractorORB->compute(img2, kpts2_new, desc2_new);

	cout << "Building kdtree1" << endl;
	vector<KeyPoint> kptsKnn1;
	vector<Point2f> ptsKnn1;
	for (int i = 0; i < kpts1_new.size(); i++) ptsKnn1.push_back(kpts1_new.at(i).pt);
	cv::flann::KDTreeIndexParams indexParams1;
	cv::flann::Index kdtree1(cv::Mat(ptsKnn1).reshape(1), indexParams1);

	cout << "Building kdtree2" << endl;
	vector<KeyPoint> kptsKnn2;
	vector<Point2f> ptsKnn2;
	for (int i = 0; i < kpts2_new.size(); i++) ptsKnn2.push_back(kpts2_new.at(i).pt);
	cv::flann::KDTreeIndexParams indexParams2;
	cv::flann::Index kdtree2(cv::Mat(ptsKnn2).reshape(1), indexParams2);

	//Match the keypoints obtained from Harris globally
	
	vector<KeyPoint> kpts1_global;
	vector<KeyPoint> kpts2_global;
	ratio_matcher_script(ratio, kpts1_new, kpts2_new, desc1_new, desc2_new, kpts1_global, kpts2_global);

	vector<KeyPoint> kpts1_global_ransac;
	vector<KeyPoint> kpts2_global_ransac;
	ransac_script(ball_radius, inlier_thr, kpts1_global, kpts2_global, homography, kpts1_global_ransac, kpts2_global_ransac);
	cout << "kpts1_global_ransac size is " << kpts1_global_ransac.size() << endl;
	cout << "kpts2_global_ransac size is " << kpts1_global_ransac.size() << endl;

	vector<KeyPoint> kpts1_all = kpts1_step3; 
	for (int i = 0; i < kpts1_global_ransac.size(); i++) kpts1_all.push_back(kpts1_global_ransac.at(i));
	vector<KeyPoint> kpts2_all = kpts2_step3;
	for (int i = 0; i < kpts2_global_ransac.size(); i++) kpts2_all.push_back(kpts2_global_ransac.at(i));

	cout << "kpts1_all size is " << kpts1_all.size() << endl;
	cout << "kpts2_all size is " << kpts2_all.size() << endl;

	//visualize the matches
	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kpts1_step3.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));
	drawMatches(img1Display, kpts1_all, img2Display, kpts2_all, matchesIndexTrivial, img1to2);
	imshow("Matches", img1to2);
	waitKey(0);



}

void test_one_nbh(std::string imagePathA, std::string imagePathB) {
	//this code take the akaze matched keypoints in two images, and tries does one nbh growth (one time)
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
	cout << "kpts1_step3 size is " << kpts1_step3.size() << endl;
	cout << "kpts2_step3 size is " << kpts2_step3.size() << endl;

	//visualize the matches
	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kpts1_step3.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));
	drawMatches(img1Display, kpts1_step3, img2Display, kpts2_step3, matchesIndexTrivial, img1to2);
	imshow("Matches", img1to2);
	waitKey(0);

	Ptr<GFTTDetector> detectorGFTT = GFTTDetector::create();
	detectorGFTT->setMaxFeatures(20000);
	detectorGFTT->setQualityLevel(0.00001);
	detectorGFTT->setMinDistance(5);
	detectorGFTT->setBlockSize(5);
	detectorGFTT->setHarrisDetector(false);
	detectorGFTT->setK(0.2);

	//Find features
	vector<KeyPoint> kpts1_new;
	detectorGFTT->detect(img1, kpts1_new);
	cout << "GFTT found " << kpts1_new.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1;
	extractorORB->compute(img1, kpts1_new, desc1);

	//Find features
	vector<KeyPoint> kpts2_new;
	detectorGFTT->detect(img2, kpts2_new);
	cout << "GFTT found " << kpts2_new.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2;
	extractorORB->compute(img2, kpts2_new, desc2);

	cout << "Building kdtree1" << endl;
	vector<KeyPoint> kptsKnn1;
	vector<Point2f> ptsKnn1;
	for (int i = 0; i < kpts1_new.size(); i++) ptsKnn1.push_back(kpts1_new.at(i).pt);
	cv::flann::KDTreeIndexParams indexParams1;
	cv::flann::Index kdtree1(cv::Mat(ptsKnn1).reshape(1), indexParams1);

	cout << "Building kdtree2" << endl;
	vector<KeyPoint> kptsKnn2;
	vector<Point2f> ptsKnn2;
	for (int i = 0; i < kpts2_new.size(); i++) ptsKnn2.push_back(kpts2_new.at(i).pt);
	cv::flann::KDTreeIndexParams indexParams2;
	cv::flann::Index kdtree2(cv::Mat(ptsKnn2).reshape(1), indexParams2);

	vector<KeyPoint> kpts1_all = kpts1_step3;
	vector<KeyPoint> kpts2_all = kpts2_step3;

	cout << "Getting nbh1" << endl;
	float radius1 = 100;
	int cardNbh1 = 100; //max number of points in the radius
	vector<int> indicesNbh1;
	vector<float> distsNbh1;
	vector<float> centerNbh1;
	centerNbh1.push_back(kpts1_step3.at(0).pt.x);
	centerNbh1.push_back(kpts1_step3.at(0).pt.y);
	kdtree1.radiusSearch(centerNbh1, indicesNbh1, distsNbh1, radius1, cardNbh1, cv::flann::SearchParams(64));
	while (!distsNbh1.empty() && (distsNbh1.back() == 0)) {
		distsNbh1.pop_back();
		indicesNbh1.pop_back();
	}
	cout << "nbh2 has size " << indicesNbh1.size() << endl;

	cout << "Getting nbh2" << endl;
	float radius2 = 100;
	int cardNbh2 = 100; //max number of points in the radius
	vector<int> indicesNbh2;
	vector<float> distsNbh2;
	vector<float> centerNbh2;
	centerNbh2.push_back(kpts2_step3.at(0).pt.x);
	centerNbh2.push_back(kpts2_step3.at(0).pt.y);
	kdtree2.radiusSearch(centerNbh2, indicesNbh2, distsNbh2, radius2, cardNbh2, cv::flann::SearchParams(64));
	while (!distsNbh2.empty() && (distsNbh2.back() == 0)) { //Guard against empty?
		distsNbh2.pop_back();
		indicesNbh2.pop_back();
	}
	cout << "nbh2 has size " << indicesNbh2.size() << endl;

	vector<KeyPoint> nbh1;
	Mat descNbh1;
	for (int i = 0; i < indicesNbh1.size(); i++) {
		int index = indicesNbh1.at(i);
		nbh1.push_back(kpts1_new.at(index));
		Mat descRow = desc1.row(index);
		descNbh1.push_back(descRow);
	}
	cout << "nbh1 has size " << nbh1.size() << endl;

	vector<KeyPoint> nbh2;
	Mat descNbh2;
	for (int i = 0; i < indicesNbh2.size(); i++) {
		int index = indicesNbh2.at(i);
		nbh2.push_back(kpts2_new.at(index));
		Mat descRow = desc2.row(index);
		descNbh2.push_back(descRow);
	}

	//Visualize the points ///VVVVVVVVVVVVVVVVVVVVVVVVVV
	namedWindow("Img1", WINDOW_AUTOSIZE);
	namedWindow("Img2", WINDOW_AUTOSIZE);
	int r = 3;
	RNG rng(12345); //random number generator

	for (size_t i = 0; i < nbh1.size(); i++) {
		drawMarker(img1Display, nbh1.at(i).pt, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), MARKER_CROSS, 10, 1);
		//circle(img1Display, nbh1.at(i).pt, r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	for (size_t i = 0; i < kpts1_step3.size(); i++) {
		circle(img1Display, kpts1_step3.at(i).pt, r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	imshow("Img1", img1Display);

	for (size_t i = 0; i < nbh2.size(); i++) {
		drawMarker(img2Display, nbh2.at(i).pt, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), MARKER_CROSS, 10, 1);
		//circle(img2Display, nbh2.at(i).pt, r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	for (size_t i = 0; i < kpts2_step3.size(); i++) {
		circle(img2Display, kpts2_step3.at(i).pt, r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	imshow("Img2", img2Display);
	waitKey(0);
	///VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

	//Match the nbhs
	vector<KeyPoint> kpts1_ratio;
	vector<KeyPoint> kpts2_ratio;
	ratio_matcher_script(0.8f, nbh1, nbh2, descNbh1, descNbh2, kpts1_ratio, kpts2_ratio);

	for (int i = 0; i < kpts1_ratio.size(); i++) {
		kpts1_all.push_back(kpts1_ratio.at(i));
		kpts2_all.push_back(kpts2_ratio.at(i));
	}
	//Add the new points to the old points

	//Draw matches
	//namedWindow("Concatenated Matching", WINDOW_AUTOSIZE);
	//visualize the matches
	Mat img1to2_concat;
	vector<DMatch> matchesIndexTrivial_concat;
	for (int i = 0; i < kpts1_all.size(); i++) matchesIndexTrivial_concat.push_back(DMatch(i, i, 0));
	drawMatches(img1Display, kpts1_all, img2Display, kpts2_all, matchesIndexTrivial_concat, img1to2_concat);
	imshow("Matches", img1to2_concat);
	waitKey(0);

}

void test_nbh_first(std::string imagePathA, std::string imagePathB) {
	//this code take the akaze matched keypoints in two images, and tries does one nbh growth (one time)
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
	cout << "kpts1_step3 size is " << kpts1_step3.size() << endl;
	cout << "kpts2_step3 size is " << kpts2_step3.size() << endl;

	//visualize the matches
	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kpts1_step3.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));
	drawMatches(img1Display, kpts1_step3, img2Display, kpts2_step3, matchesIndexTrivial, img1to2);
	imshow("Matches", img1to2);
	waitKey(0);

	Ptr<GFTTDetector> detectorGFTT = GFTTDetector::create();
	detectorGFTT->setMaxFeatures(20000);
	detectorGFTT->setQualityLevel(0.00001);
	detectorGFTT->setMinDistance(5);
	detectorGFTT->setBlockSize(5);
	detectorGFTT->setHarrisDetector(false);
	detectorGFTT->setK(0.2);

	//Find features
	vector<KeyPoint> kpts1_new;
	detectorGFTT->detect(img1, kpts1_new);
	cout << "GFTT found " << kpts1_new.size() << " feature points in image A." << endl;

	//Compute descriptors
	Ptr<DescriptorExtractor> extractorORB = ORB::create();
	Mat desc1;
	extractorORB->compute(img1, kpts1_new, desc1);

	//Find features
	vector<KeyPoint> kpts2_new;
	detectorGFTT->detect(img2, kpts2_new);
	cout << "GFTT found " << kpts2_new.size() << " feature points in image B." << endl;

	//Compute descriptors
	Mat desc2;
	extractorORB->compute(img2, kpts2_new, desc2);

	cout << "Building kdtree1" << endl;
	vector<KeyPoint> kptsKnn1;
	vector<Point2f> ptsKnn1;
	for (int i = 0; i < kpts1_new.size(); i++) ptsKnn1.push_back(kpts1_new.at(i).pt);
	cv::flann::KDTreeIndexParams indexParams1;
	cv::flann::Index kdtree1(cv::Mat(ptsKnn1).reshape(1), indexParams1);

	cout << "Building kdtree2" << endl;
	vector<KeyPoint> kptsKnn2;
	vector<Point2f> ptsKnn2;
	for (int i = 0; i < kpts2_new.size(); i++) ptsKnn2.push_back(kpts2_new.at(i).pt);
	cv::flann::KDTreeIndexParams indexParams2;
	cv::flann::Index kdtree2(cv::Mat(ptsKnn2).reshape(1), indexParams2);

	vector<KeyPoint> kpts1_all = kpts1_step3;
	vector<KeyPoint> kpts2_all = kpts2_step3;

	for (int k = 0; k < kpts1_step3.size(); k++) {
		cout << "itaration " << k << endl;
		cout << "Getting nbh1" << endl;
		float radius1 = 2000;
		int cardNbh1 = 2000; //max number of points in the radius
		vector<int> indicesNbh1;
		vector<float> distsNbh1;
		vector<float> centerNbh1;
		centerNbh1.push_back(kpts1_step3.at(k).pt.x);
		centerNbh1.push_back(kpts1_step3.at(k).pt.y);
		kdtree1.radiusSearch(centerNbh1, indicesNbh1, distsNbh1, radius1, cardNbh1, cv::flann::SearchParams(64));
		while (!distsNbh1.empty() && (distsNbh1.back() == 0)) {
			distsNbh1.pop_back();
			indicesNbh1.pop_back();
		}
		cout << "nbh2 has size " << indicesNbh1.size() << endl;

		cout << "Getting nbh2" << endl;
		float radius2 = 2000;
		int cardNbh2 = 2000; //max number of points in the radius
		vector<int> indicesNbh2;
		vector<float> distsNbh2;
		vector<float> centerNbh2;
		centerNbh2.push_back(kpts2_step3.at(k).pt.x);
		centerNbh2.push_back(kpts2_step3.at(k).pt.y);
		kdtree2.radiusSearch(centerNbh2, indicesNbh2, distsNbh2, radius2, cardNbh2, cv::flann::SearchParams(64));
		while (!distsNbh2.empty() && (distsNbh2.back() == 0)) { //Guard against empty?
			distsNbh2.pop_back();
			indicesNbh2.pop_back();
		}
		cout << "nbh2 has size " << indicesNbh2.size() << endl;

		vector<KeyPoint> nbh1;
		Mat descNbh1;
		for (int i = 0; i < indicesNbh1.size(); i++) {
			int index = indicesNbh1.at(i);
			nbh1.push_back(kpts1_new.at(index));
			Mat descRow = desc1.row(index);
			descNbh1.push_back(descRow);
		}
		cout << "nbh1 has size " << nbh1.size() << endl;

		vector<KeyPoint> nbh2;
		Mat descNbh2;
		for (int i = 0; i < indicesNbh2.size(); i++) {
			int index = indicesNbh2.at(i);
			nbh2.push_back(kpts2_new.at(index));
			Mat descRow = desc2.row(index);
			descNbh2.push_back(descRow);
		}
		vector<KeyPoint> kpts1_ratio;
		vector<KeyPoint> kpts2_ratio;
		ratio_matcher_script(0.8f, nbh1, nbh2, descNbh1, descNbh2, kpts1_ratio, kpts2_ratio);

		for (int j = 0; j < kpts1_ratio.size(); j++) kpts1_all.push_back(kpts1_ratio.at(j));
		for (int j = 0; j < kpts2_ratio.size(); j++) kpts2_all.push_back(kpts2_ratio.at(j));

		cout << "initial matching size " << kpts1_step3.size() << ", second matching size " << kpts1_ratio.size() << ", concat matching size " << kpts1_all.size() << endl;
		cout << "initial matching size " << kpts2_step3.size() << ", second matching size " << kpts2_ratio.size() << ", concat matching size " << kpts2_all.size() << endl;
	}//end for

	cout << "Getting nbh1" << endl;
	float radius1 = 100;
	int cardNbh1 = 100; //max number of points in the radius
	vector<int> indicesNbh1;
	vector<float> distsNbh1;
	vector<float> centerNbh1; 
	centerNbh1.push_back(kpts1_step3.at(0).pt.x); 
	centerNbh1.push_back(kpts1_step3.at(0).pt.y);
	kdtree1.radiusSearch(centerNbh1, indicesNbh1, distsNbh1, radius1, cardNbh1, cv::flann::SearchParams(64));
	while (!distsNbh1.empty() && (distsNbh1.back() == 0)) { 
		distsNbh1.pop_back();
		indicesNbh1.pop_back();
	}
	cout << "nbh2 has size " << indicesNbh1.size() << endl;

	cout << "Getting nbh2" << endl;
	float radius2 = 100;
	int cardNbh2 = 100; //max number of points in the radius
	vector<int> indicesNbh2;
	vector<float> distsNbh2;
	vector<float> centerNbh2;
	centerNbh2.push_back(kpts2_step3.at(0).pt.x);
	centerNbh2.push_back(kpts2_step3.at(0).pt.y);
	kdtree2.radiusSearch(centerNbh2, indicesNbh2, distsNbh2, radius2, cardNbh2, cv::flann::SearchParams(64));
	while (!distsNbh2.empty() && (distsNbh2.back() == 0)) { //Guard against empty?
		distsNbh2.pop_back();
		indicesNbh2.pop_back();
	}
	cout << "nbh2 has size " << indicesNbh2.size() << endl;

	vector<KeyPoint> nbh1;
	Mat descNbh1;
	for (int i = 0; i < indicesNbh1.size(); i++) {
		int index = indicesNbh1.at(i);
		nbh1.push_back(kpts1_new.at(index));
		Mat descRow = desc1.row(index);
		descNbh1.push_back(descRow);
	}
	cout << "nbh1 has size " << nbh1.size() << endl;

	vector<KeyPoint> nbh2;	
	Mat descNbh2;
	for (int i = 0; i < indicesNbh2.size(); i++) {
		int index = indicesNbh2.at(i);
		nbh2.push_back(kpts2_new.at(index));
		Mat descRow = desc2.row(index);
		descNbh2.push_back(descRow);
	}

	//Visualize the points ///VVVVVVVVVVVVVVVVVVVVVVVVVV
	namedWindow("Img1", WINDOW_AUTOSIZE);
	namedWindow("Img2", WINDOW_AUTOSIZE);
	int r = 3;
	RNG rng(12345); //random number generator

	for (size_t i = 0; i < nbh1.size(); i++) {
		drawMarker(img1Display, nbh1.at(i).pt, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), MARKER_CROSS, 10, 1);
		//circle(img1Display, nbh1.at(i).pt, r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	for (size_t i = 0; i < kpts1_step3.size(); i++) {
		circle(img1Display, kpts1_step3.at(i).pt, r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	imshow("Img1", img1Display);

	for (size_t i = 0; i < nbh2.size(); i++) {
		drawMarker(img2Display, nbh2.at(i).pt, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), MARKER_CROSS, 10, 1);
		//circle(img2Display, nbh2.at(i).pt, r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	for (size_t i = 0; i < kpts2_step3.size(); i++) {
		circle(img2Display, kpts2_step3.at(i).pt, r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	imshow("Img2", img2Display);
	waitKey(0);
	///VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
	
	//Match the nbhs
	vector<KeyPoint> kpts1_ratio;
	vector<KeyPoint> kpts2_ratio;
	ratio_matcher_script(0.8f, nbh1, nbh2, descNbh1, descNbh2, kpts1_ratio, kpts2_ratio);

	//Add the new points to the old points

	//Draw matches
	//namedWindow("Concatenated Matching", WINDOW_AUTOSIZE);
	//visualize the matches
	Mat img1to2_concat;
	vector<DMatch> matchesIndexTrivial_concat;
	for (int i = 0; i < kpts1_all.size(); i++) matchesIndexTrivial_concat.push_back(DMatch(i, i, 0));
	drawMatches(img1Display, kpts1_all, img2Display, kpts2_all, matchesIndexTrivial_concat, img1to2_concat);
	imshow("Matches", img1to2_concat);
	waitKey(0);

	//cout << "nbh2 has size " << nbh2.size() << endl;

	//Take a point around a keypoint in kpts1_step3
	//something does not work
	//KeyPoint trial1 = kpts1_step3.at(3);
	//KeyPoint trial2 = kpts1_step3.at(3);
	//try this
	//KeyPoint trial1 = kptsKnn1.at(3);
	//KeyPoint trial2 = kptsKnn2.at(3);

	/*
	//Find the points around these two points
	vector<KeyPoint> nbh1;
	vector<KeyPoint> nbh2;

	float radius1 = 20;
	int cardNbh1 = 100; //max number of points in the radius
	vector<int> indicesNbh1;
	vector<int> distsNbh1;
	vector<float> centerNbh1; centerNbh1.push_back(trial1.pt.x); centerNbh1.push_back(trial1.pt.y);

	float radius2 = 20;
	int cardNbh2 = 100;
	vector<int> indicesNbh2;
	vector<int> distsNbh2;
	vector<float> centerNbh2; centerNbh2.push_back(trial2.pt.x); centerNbh2.push_back(trial2.pt.y);

	//the problem is that you have to join the initial filtered keypoints for 
	kdtree1.radiusSearch(centerNbh1, indicesNbh1, distsNbh1, radius1, cardNbh1, cv::flann::SearchParams(64));
	while (!distsNbh1.empty() && (distsNbh1.back() == 0)) { //Guard against empty?
		distsNbh1.pop_back();
		indicesNbh1.pop_back();
	}

	kdtree2.radiusSearch(centerNbh2, indicesNbh2, distsNbh2, radius2, cardNbh2, cv::flann::SearchParams(64));
	while (!distsNbh2.empty() && (distsNbh2.back() == 0)) { //Guard against empty?
		distsNbh2.pop_back();
		indicesNbh2.pop_back();
	}
	*/

	/*
	vector<int> indicesKnn2;
	vector<float> distsKnn2;
	vector<float> query2; query2.push_back(ptsKnn2.at(0).x); query2.push_back(ptsKnn2.at(0).y);
	kdtree2.radiusSearch(query2, indicesKnn2, distsKnn2, 100000000, 20, cv::flann::SearchParams(64));
	while (!distsKnn2.empty() && (distsKnn2.back() == 0)) { //Guard against empty?
	distsKnn2.pop_back();
	indicesKnn2.pop_back();
	}
	for (int i = 0; i < distsKnn2.size(); i++)  kptsKnn1.push_back(kptsRatio1.at(indicesKnn1.at(i)));
	*/

}

void test_kmeans(std::string imagePathA, std::string imagePathB) {
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


	//First get points
	//Compute Descriptors
	//Do Ratio Matching
	//Do kmeans Matching
	//data = distances(kptsRatio1,kptsRatio)

	//Make an initial detection with ORB
	//Get the homography ----> pretty conservative
	//Use the homography to filter the matches before or after the kmeans,

	//Set Detector
	//Find the other detectors on page http://docs.opencv.org/trunk/d5/d51/group__features2d__main.html

	Ptr<GFTTDetector> detectorGFTT = GFTTDetector::create();
	detectorGFTT->setMaxFeatures(10000);
	detectorGFTT->setQualityLevel(0.00000001);
	detectorGFTT->setMinDistance(0.1);
	detectorGFTT->setBlockSize(5);
	detectorGFTT->setHarrisDetector(false);
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
	float ratio = 0.9f; //higher means more relaxed
	vector<KeyPoint> kptsRatio1;
	vector<KeyPoint> kptsRatio2;

	time_t tstart, tend;
	vector<vector<DMatch>> matchesLoweRatio;
	BFMatcher matcher(NORM_HAMMING);
	tstart = time(0);
	matcher.knnMatch(desc1, desc2, matchesLoweRatio, 2);

	cout << "Build kdtree" << endl;

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

	vector<int> bestLabels;
	vector<float> dataKMeans;
	for (int i = 0; i < kptsRatio1.size(); i++) {
		float distSquared = pow(kptsRatio1.at(i).pt.x - kptsRatio2.at(i).pt.x, 2) + pow(kptsRatio1.at(i).pt.y - kptsRatio2.at(i).pt.y, 2);
		dataKMeans.push_back(distSquared);
	}
	Mat centers;
	kmeans(dataKMeans, 3, bestLabels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), 5, KMEANS_PP_CENTERS, centers);
	vector<KeyPoint> kptsKmeans1;
	vector<KeyPoint> kptsKmeans2;

	for (int i = 0; i < kptsRatio1.size(); i++) {
		float d0 = centers.at<float>(0, 0);
		float d1 = centers.at<float>(1, 0);
		float d2 = centers.at<float>(2, 0);
		if (fabs(dataKMeans.at(i) - d0) < 0.3*d0 || fabs(dataKMeans.at(i) - d1) < 0.3*d1 || fabs(dataKMeans.at(i) - d2) < 0.3*d2) {
			kptsKmeans1.push_back(kptsRatio1.at(i));
			kptsKmeans2.push_back(kptsRatio2.at(i));
		}
	}

	cout << "After kmeans on dist there are " << kptsKmeans1.size() << " features. " << endl;

	//Council voting --> knn radius search dist = 1/16 of image width
	//Minimal nbh size = 3
	//
	
	vector<KeyPoint> kptsCouncil1;
	vector<KeyPoint> kptsCouncil2;

	council_filter(kptsKmeans1, kptsKmeans2, kptsCouncil1, kptsCouncil2, 30, 2);

	//End of council voting

	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kptsCouncil1.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsCouncil1, img2Display, kptsCouncil2, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);
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

	//-- Pseudo-code:
	/*
	Find features with various detectors
	Find descriptors with ORB extractor
	Do Lowe's ratio matching as initial filter, get vecKptsUnMatched1, vecKptsUnMatched2
	Build Knn trees for vecKptsUMatched1, vecKptsUMatched2
	
	#we want to find local nbh homographies
	For each point Kpt in vecKptsUnMatched1
		Find all the features within a ball of radius r centered at Kpt in vecKptsUnMatched1
		If there are more than a certain number of features the ball
			Find all the corresponding matched features in vecKptsUnMatched2
			Do RANSAC filtering with the ball of radius r in vecKptsUnMatched1 and the corresponding points in vecKptsUnMatched2, quite relaxed
			If RANSAC does not fail
				Put the matched points in vecKptsMatched1, remove then from vecKptsUnMatched1, same for image 2
				Put the rejected feature points, what remains in vecKptsUnMatched1, TWO LEVELS BELOW, 
			Else send the feature points ONE LEVEL BELOW, (RANSAC failed)
		Else remove Kpt from vecKptsUnMatched1, same for image 2
		
	vecKptsMatched1 becomes vecKptsUnMatched1L2
	vecKptsUnMatched1 becomes vecKptsUnMatched1L3
	
	Go down to a certain level and then keep vecKptsMatched1 and vecKptsMatched2
	*/


	//////////////////////////////////////////////
	/*
	Find features with various detectors
	Find descriptors with ORB extractor
	Do Lowe's ratio matching as initial filter, get vecKptsUnMatched1, vecKptsUnMatched2
	Build Knn trees for vecKptsUMatched1, vecKptsUMatched2
	*/
	//////////////////////////////////////////////

	//http://docs.opencv.org/trunk/da/d9b/group__features2d.html#ga15e1361bda978d83a2bea629b32dfd3c

	//Set Detector
	//Find the other detectors on page http://docs.opencv.org/trunk/d5/d51/group__features2d__main.html
	
	Ptr<GFTTDetector> detectorGFTT = GFTTDetector::create();
	detectorGFTT->setMaxFeatures(5000);
	detectorGFTT->setQualityLevel(0.00000001);
	detectorGFTT->setMinDistance(0.1);
	detectorGFTT->setBlockSize(5);
	detectorGFTT->setHarrisDetector(false);
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
	//-- Image 1
	vector<KeyPoint> kptsKnn1;
	vector<Point2f> ptsKnn1;
	for (int i = 0; i < kptsRatio1.size(); i++) ptsKnn1.push_back(kptsRatio1.at(i).pt);
	cv::flann::KDTreeIndexParams indexParams1;
	cv::flann::Index kdtree1(cv::Mat(ptsKnn1).reshape(1), indexParams1);

	/*
	vector<int> indicesKnn1;
	vector<float> distsKnn1;
	vector<float> query1; query1.push_back(ptsKnn1.at(0).x); query1.push_back(ptsKnn1.at(0).y); 
	kdtree1.radiusSearch(query1, indicesKnn1, distsKnn1, 100000000, 20, cv::flann::SearchParams(64));
	while (!distsKnn1.empty() && (distsKnn1.back() == 0)) { //Guard against empty?
		distsKnn1.pop_back();
		indicesKnn1.pop_back();
	}
	for (int i = 0; i < distsKnn1.size(); i++)  kptsKnn1.push_back(kptsRatio1.at(indicesKnn1.at(i)));
	*/

	//-- Image 2
	vector<KeyPoint> kptsKnn2;
	vector<Point2f> ptsKnn2;
	for (int i = 0; i < kptsRatio2.size(); i++) ptsKnn2.push_back(kptsRatio2.at(i).pt);
	cv::flann::KDTreeIndexParams indexParams2;
	cv::flann::Index kdtree2(cv::Mat(ptsKnn2).reshape(1), indexParams2);
	
	/*
	vector<int> indicesKnn2;
	vector<float> distsKnn2;
	vector<float> query2; query2.push_back(ptsKnn2.at(0).x); query2.push_back(ptsKnn2.at(0).y);
	kdtree2.radiusSearch(query2, indicesKnn2, distsKnn2, 100000000, 20, cv::flann::SearchParams(64));
	while (!distsKnn2.empty() && (distsKnn2.back() == 0)) { //Guard against empty?
		distsKnn2.pop_back();
		indicesKnn2.pop_back();
	}
	for (int i = 0; i < distsKnn2.size(); i++)  kptsKnn1.push_back(kptsRatio1.at(indicesKnn1.at(i)));
	*/

	//////////////////////////////////////////////////////////////////////////
	/*
	//we want to find local nbh homographies
		For each point Kpt in vecKptsUnMatched1
		Find all the features within a ball of radius r centered at Kpt in vecKptsUnMatched1
		If there are more than a certain number of features the ball
		Find all the corresponding matched features in vecKptsUnMatched2
		Do RANSAC filtering with the ball of radius r in vecKptsUnMatched1 and the corresponding points in vecKptsUnMatched2, quite relaxed
		If RANSAC does not fail
		Put the matched points in vecKptsMatched1, remove then from vecKptsUnMatched1, same for image 2
		Put the rejected feature points, what remains in vecKptsUnMatched1, TWO LEVELS BELOW,
		Else send the feature points ONE LEVEL BELOW, (RANSAC failed)
		Else remove Kpt from vecKptsUnMatched1, same for image 2

		vecKptsMatched1 becomes vecKptsUnMatched1L2
		vecKptsUnMatched1 becomes vecKptsUnMatched1L3

		Go down to a certain level and then keep vecKptsMatched1 and vecKptsMatched2
	*/
	/////////////////////////////////////////////////////////////////////////

	vector<vector<KeyPoint>> vecKptsUnMatchedImg1;
	vector<vector<KeyPoint>> vecKptsMatchedImg1;
	vector<vector<KeyPoint>> vecKptsUnMatchedImg2;
	vector<vector<KeyPoint>> vecKptsMatchedImg2;

	vecKptsUnMatchedImg1.push_back(kptsRatio1);
	vecKptsUnMatchedImg2.push_back(kptsRatio2);
	vector<int> radii = { 1000, 500, 100, 80, 40, 20, 10 };
	vector<int> ransacPtNbThreshhold = { 10 , 10 , 10 , 10 , 7 , 7 , 6 };
	vector<float> ransacReprojectionThreshhold = { 2000, 1000, 200, 160, 80, 40, 20 };
	int maxKpts = kptsRatio1.size();
	int nbKpts = kptsRatio1.size();
	for (int cycle = 0; cycle < 1; cycle++) {
		for (int kIdx = 0; kIdx < nbKpts; kIdx++) {
			vector<int> indicesKnn1;
			vector<float> distsKnn1;
			vector<float> query1;
			query1.push_back(kptsRatio1.at(kIdx).pt.x);
			query1.push_back(kptsRatio1.at(kIdx).pt.y);
			kdtree1.radiusSearch(query1, indicesKnn1, distsKnn1, radii.at(cycle), maxKpts, cv::flann::SearchParams(64));
			while (!distsKnn1.empty() && (distsKnn1.back() == 0)) { //Guard against empty?
				distsKnn1.pop_back();
				indicesKnn1.pop_back();
			}
			int nbhSize = indicesKnn1.size();

			//Are there enough points?
			if (nbhSize > ransacPtNbThreshhold.at(cycle)) {
				vector<KeyPoint> nbh1;
				vector<KeyPoint> nbh2;
				int nbhSize = indicesKnn1.size();
				for (int i = 0; i < nbhSize; i++) {
					nbh1.push_back(kptsRatio1.at(indicesKnn1.at(i)));
					nbh2.push_back(kptsRatio2.at(indicesKnn1.at(i))); //This is not a mistake, the indices are taken from image 1.
				}

				//-- Do RANSAC

				//Need nbh1, nbh2, indicesNbh1, indicesNbh2, indicesNbh1Pass, indicesNbh1Fail, indicesNbh2Pass, indicesNbh2Fail
				//nbh1 is nbh1
				//indicesNbh1 is indicesKnn1
				//indicesNbh2 is also indicesKnn1

				vector<int> indicesDomainPass;
				vector<int> indicesTargetPass;
				vector<int> indicesDomainFail;
				vector<int> indicesTargetFail;

				ransac_filtering(ransacReprojectionThreshhold.at(cycle), nbh1, nbh2, indicesKnn1, indicesKnn1, indicesDomainPass, indicesDomainFail, indicesTargetPass, indicesTargetFail);

				//To Do: substract the mathced points from the complete set, add to the good set
			}
		}
	}

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
			good_matches.push_back(DMatch(new_i, new_i, 0));
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

