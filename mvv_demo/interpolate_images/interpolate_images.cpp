#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>

#include "interpolate_images.h"

using namespace cv;
using namespace std;

Mat affine_transform(Vec6f sourceTri, Vec6f targetTri) {
	cv::Point2f sourceP[3];
	sourceP[0] = cv::Point2f(sourceTri[0], sourceTri[1]);
	sourceP[1] = cv::Point2f(sourceTri[2], sourceTri[3]);
	sourceP[2] = cv::Point2f(sourceTri[4], sourceTri[5]);
	cv::Point2f targetP[3];
	targetP[0] = cv::Point2f(targetTri[0], targetTri[1]);
	targetP[1] = cv::Point2f(targetTri[2], targetTri[3]);
	targetP[2] = cv::Point2f(targetTri[4], targetTri[5]);
	cv::Mat trans = cv::getAffineTransform(sourceP, targetP);
	return trans;

	/*
	Parametrization should go:

	A = [(1-t) + a_00 * t, a_01 * t,
	      a_10 * t, (1-t) + a_11 * t]
	B = [t * b_0, t * b_1]
	*/
}

vector<Mat> get_affine_transforms(vector<Vec6f> sourceT, vector<Vec6f> targetT) {
	// Start off by calculating affine transformation of two triangles.

	int numTriangles = sourceT.size();

	vector<Mat> transforms = vector<Mat>();
	for (int i = 0; i < numTriangles; i++) {
		Vec6f sourceTri = sourceT[i];
		Vec6f targetTri = targetT[i];
		cv::Point2f sourceP[3];
		sourceP[0] = cv::Point2f(sourceTri[0], sourceTri[1]);
		sourceP[1] = cv::Point2f(sourceTri[2], sourceTri[3]);
		sourceP[2] = cv::Point2f(sourceTri[4], sourceTri[5]);
		cv::Point2f targetP[3];
		targetP[0] = cv::Point2f(targetTri[0], targetTri[1]);
		targetP[1] = cv::Point2f(targetTri[2], targetTri[3]);
		targetP[2] = cv::Point2f(targetTri[4], targetTri[5]);
		cv::Mat trans = cv::getAffineTransform(sourceP, targetP);
		transforms.push_back(trans);
	}
	return transforms;
}


