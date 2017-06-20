
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
		// Convert the image to float to extract features
		Mat img1_32;
		next_1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
		Mat img2_32;
		next_2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

		Size size(1000, 600);//the dst image size,e.g.100x100
		Mat img1_32_resized;//dst image
		Mat img2_32_resized;//src image
		resize(img1_32, img1_32_resized, size);//resize image
		resize(img2_32, img2_32_resized, size);//resize image

		// Don't forget to specify image dimensions in AKAZE's options
		options.img_width = img1_32_resized.cols;
		options.img_height = img2_32_resized.rows;

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


		print_dmatches(dmatches);
	}







	return 0;
}