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
//

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
	return str.size() >= suffix.size() &&
		str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool has_image_suffix(const std::string &str) {
	return (has_suffix(str, ".jpg") || has_suffix(str, ".jpeg") || has_suffix(str, ".png") || has_suffix(str, ".bmp") || has_suffix(str, ".svg") || has_suffix(str, ".tiff") || has_suffix(str, ".ppm"));
}

int main(void)
{
	time_t ststart, tend;

	cout << "Welcome to match_points testing unit! This unit allows:" << endl;
	cout << "\t-drawing correspondances between different images," << endl;
	cout << "\t-trying different feature extraction strategies and parameters," << endl;
	cout << "\t-trying different descriptor matcher strategies and parameters," << endl;
	cout << "\t-tryint different RANSAC parameters." << endl << endl;
	
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

	string extractionType = "";
	while (true) {
		cout << endl << "Please select the feature extraction method:" << endl;
		cout << "\t1: AKAZE (approx 10 sec)" << endl;
		getline(cin, input);
		if (input == "1") {
			extractionType = input;
			break;
		}
	}

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;

	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	
	time_t tstart;
	cout << endl << "akaze->detectAndCompute(img1, noArray(), pts1, desc1)" << endl;
	tstart = time(0);
	akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
	tend = time(0);
	cout << "... took " << difftime(tend, tstart) << " s and found " << kpts1.size() << " keypoints in image 1." << endl;
	
	cout << endl << "akaze->detectAndCompute(img2, noArray(), kpts2, desc2);" << endl;
	tstart = time(0);
	akaze->detectAndCompute(img2, noArray(), kpts2, desc2);
	tend = time(0);
	cout << "... took " << difftime(tend, tstart) << " s and found " << kpts2.size() << " keypoints in image 2." << endl;

	string matcherType = "";
	while (true) {
		cout << endl << "Please select the descriptor matching method:" << endl;
		cout << "\t1: BF(Hamming Norm) with cross-check" << endl;
		cout << "\t2: BF(Hamming Norm) with Lowe's ratio test" << endl;
		getline(cin, input);
		if (input == "1" || input == "2") {
			extractionType = input;
			break;
		}
	}

	vector<KeyPoint> matchedSrc, matchedDst, inliers1, inliers2;

	vector<DMatch> matchesCrossCheck;
	if (extractionType == "1") { //cross-check
		BFMatcher matcher1(NORM_HAMMING, true); //crossCheck = true
		tstart = time(0);
		matcher1.match(desc1, desc2, matchesCrossCheck);
		int nbMatches = matchesCrossCheck.size();
		for (int i = 0; i < nbMatches; i++) {
			DMatch first = matchesCrossCheck[i];
			matchedSrc.push_back(kpts1[first.queryIdx]);
			matchedDst.push_back(kpts2[first.trainIdx]);		
		}
		tend = time(0);
		cout << "BF(NORM_HAMMING, crossCheck = true) took " << difftime(tend, tstart) << "s, " << matchesCrossCheck.size() << " matches found." << endl;
	}

	vector<vector<DMatch>> matchesLoweRatio;
	if (extractionType == "2") { //Lowe's ratio
		BFMatcher matcher2(NORM_HAMMING);
		tstart = time(0);
		matcher2.knnMatch(desc1, desc2, matchesLoweRatio, 2);
		int nbMatches = matchesLoweRatio.size();
		for (int i = 0; i < nbMatches; i++) {
			DMatch first = matchesLoweRatio[i][0];
			float dist1 = matchesLoweRatio[i][0].distance;
			float dist2 = matchesLoweRatio[i][1].distance;
			if (dist1 < nn_match_ratio * dist2) {
				matchedSrc.push_back(kpts1[first.queryIdx]);
				matchedDst.push_back(kpts2[first.trainIdx]);
			}
		}
		tend = time(0);
		cout << "BF(NORM_HAMMING, crossCheck = true):" << endl;
		cout << "\tTime taken: " << difftime(tend, tstart) << "s." << endl;
		cout << "\tRatio used: " << nn_match_ratio << endl;
		cout << "\tNumber of matches: " << matchedSrc.size() << endl;
	}

	cout << endl << "Now applying RANSAC to estimate global homography and clean up some matches." << endl;
	cout << "The RANSAC radius (max deviating distance) is " << 10 << endl;

	vector<Point2f> keysImage1;
	vector<Point2f> keysImage2;
	vector<DMatch> good_matches;

	int nbMatches = matchedSrc.size();
	for (int i = 0; i < nbMatches; i++) {
		keysImage1.push_back(matchedSrc.at(i).pt);
		keysImage2.push_back(matchedDst.at(i).pt);
	}
	float radius = 10;
	Mat H = findHomography(keysImage1, keysImage2, CV_RANSAC, radius);
	cout << "RANSAC found the homography." << endl;

	nbMatches = matchedSrc.size();
	for (int i = 0; i < nbMatches; i++) {
		Mat col = Mat::ones(3, 1, CV_64F);// , CV_32F);
		col.at<double>(0) = matchedSrc[i].pt.x;
		col.at<double>(1) = matchedSrc[i].pt.y;

		col = H * col;
		col /= col.at<double>(2); //because you are in projective space
		double dist = sqrt(pow(col.at<double>(0) - matchedDst[i].pt.x, 2) +
			pow(col.at<double>(1) - matchedDst[i].pt.y, 2));

		if (dist < inlier_threshold) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matchedSrc[i]);
			inliers2.push_back(matchedDst[i]);
			good_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}

	cout << "Filtered further the matches with the found homography with an inlier distance threshhold of " << inlier_threshold << endl;

	Mat res;
	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
	imwrite("res.png", res);

	double inlier_ratio = inliers1.size() * 1.0 / matchedSrc.size();
	cout << "A-KAZE Matching Results" << endl;
	cout << "*******************************" << endl;
	cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
	cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
	cout << "# Matches:                            \t" << matchedSrc.size() << endl;
	cout << "# Inliers:                            \t" << inliers1.size() << endl;
	cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
	cout << endl;

	cin.ignore();
	return 0;
}

void akaze_wrapper(const Mat& img_in, const float akaze_thresh, vector<KeyPoint> features_out, Mat descriptors_out) {
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	time_t tstart, tend;
	tstart = time(0);
	akaze->detectAndCompute(img_in, noArray(), features_out, descriptors_out);
	tend = time(0);
	cout << "akaze_wrapper finished:" << endl;
	cout << "\tTime = " << difftime(tstart, tend) << "s" << endl;
	cout << "\tFound " << features_out.size() << "features" << endl;
}

void ratio_matcher_wrapper(const float ratio, const vector<KeyPoint>& kpts1_in, const vector<KeyPoint>& kpts2_in, const Mat desc1_in, const Mat desc2_in, vector<KeyPoint> kpts1_out, vector<KeyPoint> kpts2_out) {
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
	cout << "BF(NORM_HAMMING, crossCheck = true):" << endl;
	cout << "\tTime taken: " << difftime(tend, tstart) << "s." << endl;
	cout << "\tRatio used: " << nn_match_ratio << endl;
	cout << "\tNumber of matches: " << kpts1_out.size() << endl;
}

void ransac_wrapper(const vector<KeyPoint>& features_in, Mat homography_out, vector<KeyPoint>& features_out) {}