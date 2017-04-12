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

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);

	tend = time(0);
	cout << "It took " << difftime(tend, tstart) << " second(s)." << endl;

	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
	vector<DMatch> good_matches;
	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;
		if (dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}
	/*
	for (unsigned i = 0; i < matched1.size(); i++) {
		Mat col = Mat::ones(3, 1, CV_32F);
		col.at<double>(0) = matched1[i].pt.x;
		col.at<double>(1) = matched1[i].pt.y;

		col = homography * col;
		col /= col.at<double>(2);
		double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
			pow(col.at<double>(1) - matched2[i].pt.y, 2));

		if (dist < inlier_threshold) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			good_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}
	*/

	for (unsigned i = 0; i < matched1.size(); i++) {
		int new_i = static_cast<int>(inliers1.size());
		inliers1.push_back(matched1[i]);
		inliers2.push_back(matched2[i]);
		good_matches.push_back(DMatch(new_i, new_i, 0));
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