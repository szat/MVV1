
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
#include <opencv2/optflow.hpp>

//#include <opencv2/features2d/features2d.hpp>
#include <AKAZE.h>
#include <AKAZEConfig.h>
//#include <cuda_profiler_api.h>
#include <opencv2/calib3d.hpp> //AKAZE seems not to work without this

using namespace std;
using namespace cv;
using namespace libAKAZECU;

int main() {
	Mat img1;
	string img1_path = "..\\data_store\\images\\c1_img_000177.png";
	img1 = imread(img1_path);
	if (img1.empty()) return -1;

	Mat img2;
	string img2_path = "..\\data_store\\images\\c2_img_000177.png";
	img2 = imread(img2_path);
	if (img2.empty()) return -1;

	//So this works well
	AKAZEOptions options;

	// Convert the image to float to extract features
	Mat img1_32;
	img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
	Mat img2_32;
	img2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

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

	return 0;
}