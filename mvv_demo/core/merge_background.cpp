#include "merge_background.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

using namespace std;
using namespace cv;

cv::Mat merge_images(cv::Mat img_1, cv::Mat img_2) {
	vector<Mat> imgs = vector<Mat>();
	imgs.push_back(img_1);
	imgs.push_back(img_2);

	Mat result;

	string name = "test1.jpg";

	Stitcher stitcher = Stitcher::createDefault();
	stitcher.stitch(imgs, result);
	return result;
}