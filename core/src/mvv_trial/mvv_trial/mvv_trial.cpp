#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/calib3d.hpp"
#include <opencv2/highgui.hpp>

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

	for (unsigned i = 0; i < matched1.size(); i++) {	
		int new_i = static_cast<int>(inliers1.size());
		inliers1.push_back(matched1[i]);
		inliers2.push_back(matched2[i]);
		good_matches.push_back(DMatch(new_i, new_i, 0));
	}

	Mat res;
	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);

	namedWindow("Result");
	imshow("Result", res);
	cvWaitKey(0);

	return 0;
}