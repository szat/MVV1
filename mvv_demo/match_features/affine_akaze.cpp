#include "affine_akaze.h"

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

void affine_skew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai)
{
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

void detect_and_compute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors)
{
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

int affine_akaze() {
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
	detect_and_compute(img1, kpts1, desc1);
	detect_and_compute(img2, kpts2, desc2);

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);

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
	imwrite("res.png", res);

	double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
	cout << "A-KAZE Matching Results" << endl;
	cout << "*******************************" << endl;
	cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
	cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
	cout << "# Matches:                            \t" << matched1.size() << endl;
	cout << "# Inliers (fake):                            \t" << inliers1.size() << endl;
	cout << "# Inliers Ratio (fake):                      \t" << inlier_ratio << endl;
	cout << endl;


	cout << M_PI;
	cin.ignore();
	return 0;
}
