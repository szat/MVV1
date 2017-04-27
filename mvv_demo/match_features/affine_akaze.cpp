#include "affine_akaze.h"
#include "match_points.h"

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

#define M_PI 3.14159265358979323846

using namespace std;
using namespace cv;
const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
const double akaze_thresh = 3e-4;    // AKAZE detection threshold set to locate about 1000 keypoints

void affine_skew_here(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai)
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

void detect_and_compute_here(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors)
{
	cout << "Applying detect_and_compute(img, keypoints, descriptors)..." << endl;

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

			affine_skew_here(t, phi, timg, mask, Ai);

#if 0
			Mat img_disp;
			bitwise_and(mask, timg, img_disp);
			namedWindow("Skew", WINDOW_AUTOSIZE);// Create a window for display.
			imshow("Skew", img_disp);
			waitKey(0);
#endif
			const double akaze_thresh = 3e-4;    // AKAZE detection threshold set to locate about 1000 keypoints

			Ptr<AKAZE> akaze = AKAZE::create();
			akaze->setThreshold(akaze_thresh);

			akaze->detect(timg, kps, mask);
			akaze->compute(timg, kps, desc);

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

void affine_akaze_test(string imagePathA_in, string imagePathB_in, vector<KeyPoint>& keysImgA_out, vector<KeyPoint>& keysImgB_out) {
	Mat img1 = imread(imagePathA_in, IMREAD_GRAYSCALE);
	if (!img1.data) { 
		cout << "Error loading " << imagePathA_in << endl;
		return; }
	Mat img1Display = imread(imagePathA_in);
	Mat img2 = imread(imagePathB_in, IMREAD_GRAYSCALE);
	if (!img2.data) {
		cout << "Error loading " << imagePathB_in << endl;
		return;
	}
	Mat img2Display = imread(imagePathB_in);

	cout << "Starting affine_akaze(" << imagePathA_in <<" , " << imagePathB_in << ") ..." << endl;

	vector<KeyPoint> kptsImg1, kptsImg2;
	Mat descImg1, descImg2;
	detect_and_compute_here(img1, kptsImg1, descImg1);
	detect_and_compute_here(img2, kptsImg2, descImg2);

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(descImg1, descImg2, nn_matches, 2);

	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
	vector<DMatch> good_matches;
	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;
		if (dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kptsImg1[first.queryIdx]);
			matched2.push_back(kptsImg2[first.trainIdx]);
		}
	}

	for (unsigned i = 0; i < matched1.size(); i++) {
		int new_i = static_cast<int>(inliers1.size());
		inliers1.push_back(matched1[i]);
		inliers2.push_back(matched2[i]);
		good_matches.push_back(DMatch(new_i, new_i, 0));
	}

	keysImgA_out = inliers1;
	keysImgB_out = inliers2;
	vector<KeyPoint> kptsRansac1;
	vector<KeyPoint> kptsRansac2;
	vector<int> indicesDomain;
	vector<int> indicesTarget;
	vector<int> indicesDomainPass;
	vector<int> indicesTargetPass;
	vector<int> indicesDomainFail;
	vector<int> indicesTargetFail;

	Mat img1to2;
	vector<DMatch> matchesIndexTrivial;
	for (int i = 0; i < kptsRansac1.size(); i++) matchesIndexTrivial.push_back(DMatch(i, i, 0));

	drawMatches(img1Display, kptsRansac1, img2Display, kptsRansac2, matchesIndexTrivial, img1to2);

	//-- Show detected matches
	imshow("Matches", img1to2);

	waitKey(0);
	
}
