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
#include <opencv2/videoio.hpp>

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
	/*
	int starter_offset = 10;
	// camera 1, flash test 185
	// camera 2, flash test 410
	string path_1 = "flash_test_1.MP4";
	string path_2 = "flash_test_2.MP4";

	int start_1 = 185 + starter_offset;
	int start_2 = 410 + starter_offset;

	VideoCapture cap_1(path_1);
	VideoCapture cap_2(path_2);

	if (!cap_1.isOpened()) {
		cout << "Hey there" << endl;
	}
	if (!cap_2.isOpened()) {
		cout << "Hey there" << endl;
	}

	Mat img11;
	Mat img22;

	cap_1.set(CAP_PROP_POS_FRAMES, 185+10);
	cap_2.set(CAP_PROP_POS_FRAMES, 410+10);

	cap_1.read(img11);
	cap_2.read(img22);
	
	imwrite("../data_store/images/frame1.png", img11);
	imwrite("../data_store/images/frame2.png", img22);
	*/
	Mat img_1;
	string img1_path = "../data_store/images/c1.png";
	img_1 = imread(img1_path);
	if (img_1.empty()) return -1;

	Mat img_2;
	string img2_path = "../data_store/images/c2.png";
	img_2 = imread(img2_path);
	if (img_2.empty()) return -1;

	Mat img_1_test;
	string img1_test_path = "../data_store/images/frame1.jpg";
	img_1_test = imread(img1_test_path);
	if (img_1_test.empty()) return -1;

	Mat img_2_test;
	string img2_test_path = "../data_store/images/frame2.jpg";
	img_2_test = imread(img2_test_path);
	if (img_2_test.empty()) return -1;

	Mat img1r;//dst image
	Mat img2r;//src image
	resize(img_1_test, img1r, Size(812,612));//resize image
	resize(img_2_test, img2r, Size(812,612));//resize image

	Mat img3 = img1r.clone();
	Mat img4 = img2r.clone();

	cout << (img_1.isContinuous() ? "true" : "false") << endl;
	cout << (img_2.isContinuous() ? "true" : "false") << endl;
	cout << (img_1_test.isContinuous() ? "true" : "false") << endl;
	cout << (img_2_test.isContinuous() ? "true" : "false") << endl;
	cout << (img3.isContinuous() ? "true" : "false") << endl;
	cout << (img4.isContinuous() ? "true" : "false") << endl;

	//So this works well
	AKAZEOptions options;

	// Convert the image to float to extract features
	Mat img1_32;
	img_1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
	Mat img2_32;
	img_2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

	// Don't forget to specify image dimensions in AKAZE's options
	options.img_width = img1_32.cols;
	options.img_height = img1_32.rows;

	// Extract features
	libAKAZECU::AKAZE evolution(options);
	vector<KeyPoint> kpts1;
	vector<KeyPoint> kpts2;
	vector<vector<cv::DMatch> > dmatches;
	Mat desc1;
	Mat desc2;

	if (img1_32.isContinuous()) {
		cout << "img1_32 is continuous" << endl;
	}
	else {
		cout << "img1_32 is not continuous" << endl;
	}

	if (img2_32.isContinuous()) {
		cout << "img2_32 is continuous" << endl;
	}
	else {
		cout << "img2_32 is not continuous" << endl;
	}

	evolution.Create_Nonlinear_Scale_Space(img1_32);
	evolution.Feature_Detection(kpts1);
	evolution.Compute_Descriptors(kpts1, desc1);

	evolution.Create_Nonlinear_Scale_Space(img2_32);
	evolution.Feature_Detection(kpts2);
	evolution.Compute_Descriptors(kpts2, desc2);

	Matcher cuda_matcher;

	cuda_matcher.bfmatch(desc1, desc2, dmatches);
	cuda_matcher.bfmatch(desc2, desc1, dmatches);

	std::cout << "#matches: " << dmatches.size() << std::endl;
	std::cout << "#kptsq:   " << kpts1.size() << std::endl;
	std::cout << "#kptst:   " << kpts2.size() << std::endl;

	vector<cv::Point2f> matches, inliers;
	matches2points_nndr(kpts2, kpts1, dmatches, matches, DRATIO);
	compute_inliers_ransac(matches, inliers, MIN_H_ERROR, false);

	// Prepare the visualization

	Mat img_com;

	draw_keypoints(img1r, kpts1);
	draw_keypoints(img2r, kpts2);
	draw_inliers(img1r, img2r, img_com, inliers);
	cv::namedWindow("Inliers", cv::WINDOW_NORMAL);
	cv::imshow("Inliers", img_com);
	cv::waitKey(0);

	return 0;
}

/*
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
#include <opencv2/videoio.hpp>

//#include <opencv2/features2d/features2d.hpp>
#include <AKAZE.h>
#include <AKAZEConfig.h>
//#include <cuda_profiler_api.h>
#include <opencv2/calib3d.hpp> //AKAZE seems not to work without this

using namespace std;
using namespace cv;
using namespace libAKAZECU;

void print_dmatches(vector<vector<DMatch>> dmatches) {
	cout << dmatches.size() << endl;
	cout << dmatches[0].size() << endl;
}

int main() {
	int starter_offset = 10;
	// camera 1, flash test 185
	// camera 2, flash test 410
	string path_1 = "../data_store/flash/flash_test_1.MP4";
	string path_2 = "../data_store/flash/flash_test_2.MP4";

	int start_1 = 185 + starter_offset;
	int start_2 = 410 + starter_offset;

	VideoCapture cap_1(path_1);
	VideoCapture cap_2(path_2);

	if (!cap_1.isOpened()) {
		cout << "Hey there" << endl;
	}
	if (!cap_2.isOpened()) {
		cout << "Hey there" << endl;
	}

	int num_frames_1 = cap_1.get(CAP_PROP_FRAME_COUNT);
	int num_frames_2 = cap_2.get(CAP_PROP_FRAME_COUNT);

	cap_1.set(CV_CAP_PROP_POS_FRAMES, start_1);
	cap_2.set(CV_CAP_PROP_POS_FRAMES, start_2);

	Mat next_1;
	Mat next_2;

	//So this works well
	AKAZEOptions options;

	for (int i = 0; i < 1; i++) {
		cap_1.read(next_1);
		cap_2.read(next_2);

		next_1 = imread("..\\data_store\\images\\david_1.jpg");
		next_2 = imread("..\\data_store\\images\\david_2.jpg");

		// Convert the image to float to extract features
		Mat img1_32;
		next_1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
		Mat img2_32;
		next_2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

		Size size(1000, 600);//the dst image size,e.g.100x100
		Mat img1_32_resized;//dst image
		Mat img2_32_resized;//src image
		//resize(img1_32, img1_32_resized, size);//resize image
		//resize(img2_32, img2_32_resized, size);//resize image

		// Don't forget to specify image dimensions in AKAZE's options
<<<<<<< HEAD
		options.img_width = img1_32_resized.cols;
		options.img_height = img1_32_resized.rows;
=======
		options.img_width = img1_32.cols;
		options.img_height = img2_32.rows;
>>>>>>> 987824817a9baea5e01eabc92dcf02ad327c52b6

		// Extract features
		libAKAZECU::AKAZE evolution(options);
		vector<KeyPoint> kpts1;
		vector<KeyPoint> kpts2;
		vector<vector<cv::DMatch> > dmatches;
		Mat desc1;
		Mat desc2;

		evolution.Create_Nonlinear_Scale_Space(img1_32_resized);
		evolution.Feature_Detection(kpts1);
		evolution.Compute_Descriptors(kpts1, desc1);

		evolution.Create_Nonlinear_Scale_Space(img2_32_resized);
		evolution.Feature_Detection(kpts2);
		evolution.Compute_Descriptors(kpts2, desc2);

		Matcher cuda_matcher;

		cuda_matcher.bfmatch(desc1, desc2, dmatches);
		cuda_matcher.bfmatch(desc2, desc1, dmatches);


		print_dmatches(dmatches);
	}







	return 0;
}

*/