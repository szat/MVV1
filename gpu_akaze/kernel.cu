
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/video.hpp>
//#include <opencv2/optflow.hpp>

//#include <opencv2/features2d/features2d.hpp>
#include <AKAZE.h>
#include <AKAZEConfig.h>
//#include <cuda_profiler_api.h>
#include <opencv2/calib3d.hpp> //AKAZE seems not to work without this

using namespace std;
using namespace cv;
using namespace libAKAZECU;

const float MIN_H_ERROR = 5.00f;            ///< Maximum error in pixels to accept an inlier
const float DRATIO = 0.80f;

int main() {
	VideoCapture cap_1("..\\data_store\\judo\\judo_left.MP4");
	if (!cap_1.isOpened()) {
		cout << "Video 1 failed to load." << endl;
		return -1;
	}

	VideoCapture cap_2("..\\data_store\\judo\\judo_right.MP4");
	if (!cap_2.isOpened()) {
		cout << "Video 1 failed to load." << endl;
		return -1;
	}

	int num_frames_1 = cap_1.get(CAP_PROP_FRAME_COUNT);
	int num_frames_2 = cap_2.get(CAP_PROP_FRAME_COUNT);

	int start = 500;
	int offset = 595;
	cap_1.set(CV_CAP_PROP_POS_FRAMES, start+offset);
	cap_2.set(CV_CAP_PROP_POS_FRAMES, start);

	Mat img1;
	cap_1.read(img1);
	img1 = img1.clone();
	Mat img2;
	cap_2.read(img2);
	img2 = img2.clone();

	imwrite("..\\data_store\\images\\judo_left.png", img1);
	imwrite("..\\data_store\\images\\judo_right.png", img2);

	//So this works well
	AKAZEOptions options;

	// Convert the image to float to extract features
	Mat img1_gray;
	cvtColor(img1, img1_gray, CV_BGR2GRAY);
	Mat img2_gray;
	cvtColor(img2, img2_gray, CV_BGR2GRAY);
	Mat img1_32;
	img1_gray.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
	Mat img2_32;
	img2_gray.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

	// Don't forget to specify image dimensions in AKAZE's options
	options.img_width = img1.cols;
	options.img_height = img1.rows;

	// Extract features
	libAKAZECU::AKAZE evolution(options);
	vector<KeyPoint> kpts1;
	vector<KeyPoint> kpts2;
	vector<vector<cv::DMatch> > dmatches;
	Mat desc1;
	Mat desc2;

	evolution.Create_Nonlinear_Scale_Space(img1_32);
	evolution.Feature_Detection(kpts1);
	evolution.Compute_Descriptors(kpts1, desc1);

	evolution.Create_Nonlinear_Scale_Space(img2_32);
	evolution.Feature_Detection(kpts2);
	evolution.Compute_Descriptors(kpts2, desc2);

	Matcher cuda_matcher;

	cuda_matcher.bfmatch(desc1, desc2, dmatches);
	cuda_matcher.bfmatch(desc2, desc1, dmatches);

	vector<cv::Point2f> matches, inliers;

	matches2points_nndr(kpts2, kpts1, dmatches, matches, DRATIO);
	compute_inliers_ransac(matches, inliers, MIN_H_ERROR, false);

	Mat img_com = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);
	draw_keypoints(img1, kpts1);
	draw_keypoints(img2, kpts2);
	draw_inliers(img1, img2, img_com, inliers);
	cv::namedWindow("Inliers", cv::WINDOW_NORMAL);
	cv::imshow("Inliers", img_com);
	cv::waitKey(0);

	return 0;
}