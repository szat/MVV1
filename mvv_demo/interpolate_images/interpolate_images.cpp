#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>

#include "interpolate_images.h"

using namespace cv;
using namespace std;

static void onChangeTriangleMorph(int morph, void *userdata) //void* mean that it is a pointer of unknown type
{
	(*((trackbarTriangleMorph*)userdata)).morph = morph;

	vector<KeyPoint> srcPoints = (*((trackbarTriangleMorph*)userdata)).sourcePoints;
	vector<KeyPoint> tarPoints = (*((trackbarTriangleMorph*)userdata)).targetPoints;
	Rect imgSize = (*((trackbarTriangleMorph*)userdata)).imageSize;
	vector<Point2f> interPoints = construct_intermediate_points(srcPoints, tarPoints, morph);
	Subdiv2D subdiv = raw_triangulation(interPoints, imgSize);
	display_triangulation(subdiv, imgSize);
}

int interpolation_trackbar(vector<Vec6f> trianglesA, vector<Vec6f> trianglesB, Rect imgSizeA, Rect imageSizeB, vector<vector<double>> )
{
	trackbarTriangleMorph holder;
	holder.sourcePoints = sourcePoints;
	holder.targetPoints = targetPoints;
	holder.imageSize = imgSize;
	holder.morph = 0;

	int morph = 0;

	namedWindow("Adjust Window");
	cvCreateTrackbar2("Morph", "Adjust Window", &morph, 100, onChangeTriangleMorph, (void*)(&holder));
	waitKey(0);

	return 0;
}

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


vector<vector<double>> interpolation_preprocessing(vector<Vec6f> sourceT, vector<Vec6f> targetT) {
	int numTriangles = sourceT.size();
	vector<Mat> transforms = get_affine_transforms(sourceT, targetT);
	// convert to a more readable form
	vector<vector<double>> interpolation_params = vector<vector<double>>();

	for (int i = 0; i < numTriangles; i++) {
		vector<double> currentParamsNumeric = vector<double>();
		Mat currentParams = transforms[i];
		// a00, a01, b00, a10, a11, b01
		currentParamsNumeric.push_back(currentParams.at<double>(0, 0));
		currentParamsNumeric.push_back(currentParams.at<double>(0, 1));
		currentParamsNumeric.push_back(currentParams.at<double>(0, 2));
		currentParamsNumeric.push_back(currentParams.at<double>(1, 0));
		currentParamsNumeric.push_back(currentParams.at<double>(1, 1));
		currentParamsNumeric.push_back(currentParams.at<double>(1, 2));
		interpolation_params.push_back(currentParamsNumeric);
	}

	return interpolation_params;
}