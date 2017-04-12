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
const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
const double akaze_thresh = 3e-4;    // AKAZE detection threshold set to locate about 1000 keypoints

int main(void)
{
	Mat img1 = imread("..\\data_store\\mona_lisa_1.jpg", IMREAD_GRAYSCALE);
	Mat img2 = imread("..\\data_store\\mona_lisa_4.jpg", IMREAD_GRAYSCALE);
	//Mat img2 = imread("..\\data_store\\mona_lisa_3.jpg", IMREAD_GRAYSCALE);
	//Mat img2 = imread("..\\data_store\\mona_lisa_2.jpg", IMREAD_GRAYSCALE);
	//Mat img1 = imread("..\\data_store\\david_1.jpg", IMREAD_GRAYSCALE);
	//Mat img2 = imread("..\\data_store\\david_2.jpg", IMREAD_GRAYSCALE);
	//Mat img1 = imread("..\\data_store\\arc_de_triomphe_1.png", IMREAD_GRAYSCALE);
	//Mat img2 = imread("..\\data_store\\arc_de_triomphe_2.png", IMREAD_GRAYSCALE);

	float homography_entries[9] = { 7.6285898e-01, -2.9922929e-01,   2.2567123e+02,
		3.3443473e-01,  1.0143901e+00, -7.6999973e+01,
		3.4663091e-04, -1.4364524e-05,  1.0000000e+00 };
	Mat homography = Mat(3, 3, CV_32F, homography_entries);
	Mat warped_image;
	warpPerspective(img1, warped_image, homography, img1.size());
	img2 = warped_image;

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);

	time_t tstart, tend;
	tstart = time(0);

	akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

	//Test different matching methods
	vector<DMatch> matches1;
	BFMatcher matcher1(NORM_HAMMING, true); //crossCheck = true
	tstart = time(0);
	matcher1.match(desc1, desc2, matches1);
	tend = time(0);
	cout << "Brute force matching with cross-check took " << difftime(tend, tstart) << " second(s)." << endl;
	cout << "Number of initial matches (outliers and inliers) " << matches1.size() << endl;

	vector<vector<DMatch>> matches2;
	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
	vector<DMatch> good_matches;
	
	BFMatcher matcher2(NORM_HAMMING);
	tstart = time(0);
	matcher2.knnMatch(desc1, desc2, matches2, 2);
	int nbMatches = matches2.size();
	for (size_t i = 0; i < nbMatches; i++) {
		DMatch first = matches2[i][0];
		float dist1 = matches2[i][0].distance;
		float dist2 = matches2[i][1].distance;
		if (dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}
	tend = time(0);
	cout << "Brute force matching with Lowe's ratio test took " << difftime(tend, tstart) << " second(s)." << endl;
	cout << "Number of initial matches (outliers and inliers) " << matched2.size() << endl;

	vector<Point2f> keysImage1;
	vector<Point2f> keysImage2;
	for (auto & element : matched1) keysImage1.push_back(element.pt);
	for (auto & element : matched2) keysImage2.push_back(element.pt);
	Mat H = findHomography(keysImage1, keysImage2, CV_RANSAC);

	for (size_t i = 0; i < matched1.size(); i++) {
		Mat col = Mat::ones(3, 1, CV_64F);// , CV_32F);
		cout << matched1[i].pt.x << endl;
		col.at<double>(0) = matched1[i].pt.x;
		col.at<double>(1) = matched1[i].pt.y;

		col = H * col;
		col /= col.at<double>(2); //because you are in projective space
		double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
			pow(col.at<double>(1) - matched2[i].pt.y, 2));

		if (dist < inlier_threshold) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			good_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}

	Mat res;
	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
	imwrite("res.png", res);

	double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
	cout << "A-KAZE Matching Results" << endl;
	cout << "*******************************" << endl;
	cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
	cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
	cout << "# Matches:                            \t" << matched1.size() << endl;
	cout << "# Inliers:                            \t" << inliers1.size() << endl;
	cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
	cout << endl;

	cin.ignore();
	return 0;
}