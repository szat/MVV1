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

vector<Vec6f> get_interpolated_triangles(vector<Vec6f> sourceT, vector<Vec6f> targetT, vector<vector<double>> affine, int tInt) {
	int numTriangles = sourceT.size();
	float t = (float)tInt / 100;
	vector<Vec6f> interT = vector<Vec6f>();

	for (int i = 0; i < numTriangles; i++) {
		vector<double> affineParams = affine[i];
		float pt1x = (1 - t + affineParams[0] * t) * sourceT[i][0] + affineParams[1] * sourceT[i][1] + affineParams[2];
		float pt1y = (affineParams[3] * t) * sourceT[i][0] + affineParams[4] * sourceT[i][1] + affineParams[5];
		float pt2x = (1 - t + affineParams[0] * t) * sourceT[i][2] + affineParams[1] * sourceT[i][3] + affineParams[2];
		float pt2y = (affineParams[3] * t) * sourceT[i][2] + affineParams[4] * sourceT[i][3] + affineParams[5];
		float pt3x = (1 - t + affineParams[0] * t) * sourceT[i][4] + affineParams[1] * sourceT[i][5] + affineParams[2];
		float pt3y = (affineParams[3] * t) * sourceT[i][4] + affineParams[4] * sourceT[i][5] + affineParams[5];
		interT.push_back(Vec6f(pt1x, pt1y, pt2x, pt2y, pt3x, pt3y));
	}
	return interT;
}

void display_interpolated_triangles(vector<Vec6f> triangles, Rect imageBounds) {
	// the graphical_triangulation function is far too slow

	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Mat img(imageBounds.size(), CV_8UC3);

	img = Scalar::all(0);
	string win = "Delaunay Demo";

	int numTriangles = triangles.size();
	for (size_t i = 0; i < numTriangles; i++)
	{
		Vec6f t = triangles[i];
		Point pt0 = Point(cvRound(t[0]), cvRound(t[1]));
		Point pt1 = Point(cvRound(t[2]), cvRound(t[3]));
		Point pt2 = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt0, pt1, active_facet_color, 1, LINE_AA, 0);
		line(img, pt1, pt2, active_facet_color, 1, LINE_AA, 0);
		line(img, pt2, pt0, active_facet_color, 1, LINE_AA, 0);
	}

	imshow(win, img);
	waitKey(1);
}

struct interpolationMorph {
	Rect imageSize;
	int tInt;
	vector<Vec6f> sourceT;
	vector<Vec6f> targetT;
	vector<vector<double>> affineParams;
};

static void onInterpolate(int tInt, void *userdata) //void* mean that it is a pointer of unknown type
{
	(*((interpolationMorph*)userdata)).tInt = tInt;

	vector<Vec6f> sourceT = (*((interpolationMorph*)userdata)).sourceT;
	vector<Vec6f> targetT = (*((interpolationMorph*)userdata)).targetT;
	vector<vector<double>> affineParams = (*((interpolationMorph*)userdata)).affineParams;
	Rect imgSize = (*((interpolationMorph*)userdata)).imageSize;
	vector<Vec6f> interT = get_interpolated_triangles(sourceT, targetT, affineParams, tInt);
	display_interpolated_triangles(interT, imgSize);
	//Subdiv2D subdiv = raw_triangulation(interPoints, imgSize);
	//display_triangulation(subdiv, imgSize);
}

int interpolation_trackbar(vector<Vec6f> trianglesA, vector<Vec6f> trianglesB, Rect imgSizeA, Rect imgSizeB, vector<vector<double>> affine)
{
	// max of height and weidth
	int maxWidth = max(imgSizeA.width, imgSizeB.width);
	int maxHeight = max(imgSizeA.height, imgSizeB.height);
	Rect imgSize = Rect(0,0,maxWidth, maxHeight);

	interpolationMorph holder;
	holder.sourceT = trianglesA;
	holder.targetT = trianglesB;
	holder.imageSize = imgSize;
	holder.affineParams = affine;
	holder.tInt = 0;

	int tInt = 0;

	namedWindow("Adjust Window");
	cvCreateTrackbar2("Morph", "Adjust Window", &tInt, 100, onInterpolate, (void*)(&holder));
	waitKey(0);

	return 0;
}
