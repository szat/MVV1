#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>
#include <algorithm> 

#include "interpolate_images.h"

#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)

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

vector<Vec6f> get_interpolated_triangles(vector<Vec6f> sourceT, vector<Vec6f> targetT, vector<vector<vector<double>>> affine, int tInt) {
	int numTriangles = sourceT.size();
	float t = (float)tInt / 100;
	vector<Vec6f> interT = vector<Vec6f>();

	// It is by will alone I set my mind in motion.
	// Good luck with these indices...
	for (int i = 0; i < numTriangles; i++) {
		vector<vector<double>> affineParams = affine[i];
		float pt1x = (1 - t + affineParams[0][0] * t) * sourceT[i][0] + (affineParams[0][1] * t) * sourceT[i][1] + (affineParams[0][2] * t);
		float pt1y = (affineParams[1][0] * t) * sourceT[i][0] + (1 - t + affineParams[1][1] * t) * sourceT[i][1] + (affineParams[1][2] * t);
		float pt2x = (1 - t + affineParams[0][0] * t) * sourceT[i][2] + (affineParams[0][1] * t) * sourceT[i][3] + (affineParams[0][2] * t);
		float pt2y = (affineParams[1][0] * t) * sourceT[i][2] + (1 - t + affineParams[1][1] * t) * sourceT[i][3] + (affineParams[1][2] * t);
		float pt3x = (1 - t + affineParams[0][0] * t) * sourceT[i][4] + (affineParams[0][1] * t) * sourceT[i][5] + (affineParams[0][2] * t);
		float pt3y = (affineParams[1][0] * t) * sourceT[i][4] + (1 - t + affineParams[1][1] * t) * sourceT[i][5] + (affineParams[1][2] * t);
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
	vector<vector<vector<double>>> affineParams;
};

static void onInterpolate(int tInt, void *userdata) //void* mean that it is a pointer of unknown type
{
	(*((interpolationMorph*)userdata)).tInt = tInt;

	vector<Vec6f> sourceT = (*((interpolationMorph*)userdata)).sourceT;
	vector<Vec6f> targetT = (*((interpolationMorph*)userdata)).targetT;
	vector<vector<vector<double>>> affineParams = (*((interpolationMorph*)userdata)).affineParams;
	Rect imgSize = (*((interpolationMorph*)userdata)).imageSize;
	vector<Vec6f> interT = get_interpolated_triangles(sourceT, targetT, affineParams, tInt);
	display_interpolated_triangles(interT, imgSize);
	//Subdiv2D subdiv = raw_triangulation(interPoints, imgSize);
	//display_triangulation(subdiv, imgSize);
}

void interpolation_trackbar(vector<Vec6f> trianglesA, vector<Vec6f> trianglesB, Rect imgSizeA, Rect imgSizeB, vector<vector<vector<double>>> affine)
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
}

Mat get_affine_intermediate(Mat affine, float t) {
	// parametrization
	double a00 = 1 - t + t * affine.at<double>(0, 0);
	double a01 = t * affine.at<double>(0, 1);
	double b00 = t * affine.at<double>(0, 2);
	double a10 = t * affine.at<double>(1, 0);
	double a11 = 1 - t + t * affine.at<double>(1, 1);
	double b01 = t * affine.at<double>(1, 2);
	Mat affineIntermediate = (Mat_<double>(2, 3) << a00, a01, b00, a10, a11, b01);
	return affineIntermediate;
}

void purple_mesh_test() {
	int xDim = 600;
	int yDim = 600;
	Mat imgRed(xDim, yDim, CV_8UC3, Scalar(255, 0, 0));
	Mat imgBlue(xDim, yDim, CV_8UC3, Scalar(0, 0, 255));
	/*
	imshow("red", imgRed);
	waitKey(1);
	imshow("blue", imgBlue);
	waitKey(1);
	*/

	// use masks
	float t = 0.5;

	Vec6f triA_1 = Vec6f(10, 10, 40, 45, 21, 125);
	Vec6f triB_1 = Vec6f(201, 242, 240, 255, 201, 293);
	Vec6f triA_2 = Vec6f(300, 300, 400, 400, 400, 500);
	Vec6f triB_2 = Vec6f(298, 311, 391, 421, 410, 550);
	// get affine
	Mat affineA = affine_transform(triA_1, triB_1);
	Mat affineReverseA = affine_transform(triB_1, triA_1);
	Mat affinePartialA = get_affine_intermediate(affineA, t);
	Mat affineB = affine_transform(triA_2, triB_2);
	Mat affineReverseB = affine_transform(triB_2, triA_2);
	Mat affinePartialB = get_affine_intermediate(affineB, t);
	Mat maskA_1 = cv::Mat::zeros(xDim, yDim, CV_8U);
	Mat maskB_1 = cv::Mat::zeros(xDim, yDim, CV_8U);
	Mat maskA_2 = cv::Mat::zeros(xDim, yDim, CV_8U);
	Mat maskB_2 = cv::Mat::zeros(xDim, yDim, CV_8U);

	Point ptsA_1[3] = {
		Point(triA_1[0],triA_1[1]),
		Point(triA_1[2],triA_1[3]),
		Point(triA_1[4],triA_1[5]),
	};
	Point ptsB_1[3] = {
		Point(triB_1[0],triB_1[1]),
		Point(triB_1[2],triB_1[3]),
		Point(triB_1[4],triB_1[5]),
	};
	Point ptsA_2[3] = {
		Point(triA_2[0],triA_2[1]),
		Point(triA_2[2],triA_2[3]),
		Point(triA_2[4],triA_2[5]),
	};
	Point ptsB_2[3] = {
		Point(triB_2[0],triB_2[1]),
		Point(triB_2[2],triB_2[3]),
		Point(triB_2[4],triB_2[5]),
	};

	fillConvexPoly(maskA_1, ptsA_1, 3, Scalar(1));
	fillConvexPoly(maskB_1, ptsB_1, 3, Scalar(1));
	fillConvexPoly(maskA_2, ptsA_2, 3, Scalar(1));
	fillConvexPoly(maskB_2, ptsB_2, 3, Scalar(1));

	Mat tempImgA = Mat::zeros(xDim, yDim, CV_8U);
	Mat tempImgB = Mat::zeros(xDim, yDim, CV_8U);
	Mat dst = Mat::zeros(xDim, yDim, CV_8U);
	
	imgRed.copyTo(tempImgA, maskA_1);
	imgBlue.copyTo(tempImgB, maskB_1);

	warpAffine(tempImgA, tempImgA, affinePartialA, Size(xDim, yDim));
	warpAffine(tempImgB, tempImgB, affineReverseA, Size(xDim, yDim));
	warpAffine(tempImgB, tempImgB, affinePartialA, Size(xDim, yDim));

	addWeighted(tempImgA, t, tempImgB, 1 - t, 0.0, dst);
	
	
	Mat dst2 = Mat::zeros(xDim, yDim, CV_8U);
	tempImgA = Mat::zeros(xDim, yDim, CV_8U);
	tempImgB = Mat::zeros(xDim, yDim, CV_8U);
	imgRed.copyTo(tempImgA, maskA_2);
	imgBlue.copyTo(tempImgB, maskB_2);

	warpAffine(tempImgA, tempImgA, affinePartialB, Size(xDim, yDim));
	warpAffine(tempImgB, tempImgB, affineReverseB, Size(xDim, yDim));
	warpAffine(tempImgB, tempImgB, affinePartialB, Size(xDim, yDim));

	addWeighted(tempImgA, t, tempImgB, 1 - t, 0.0, dst2);

	addWeighted(dst, 1, dst2, 1, 0.0, dst);
	
	imshow("purple", dst);
	waitKey(1);


	cout << "Purple mesh test";
}

void fill_bottom_flat_triangle(vector<Point2f> &points, Point2f v1, Point2f v2, Point2f v3)
{
	Point2f temp;
	if (v2.x >= v3.x) {
		temp = v2;
		v2 = v3;
		v3 = temp;
	}
	float invslope1 = (v2.x - v1.x) / (v2.y - v1.y);
	float invslope2 = (v3.x - v1.x) / (v3.y - v1.y);
	float curx1 = v1.x;
	float curx2 = v1.x;
	for (int scanlineY = (int)v1.y; scanlineY >= v2.y; scanlineY--)
	{
		for (int scanlineX = (int)curx1; scanlineX <= (int)curx2; scanlineX++) {
			points.push_back(Point2f(scanlineX, scanlineY));
		}
		curx1 -= invslope1;
		curx2 -= invslope2;
	}
}

void fill_top_flat_triangle(vector<Point2f> &points, Point2f v1, Point2f v2, Point2f v3)
{
	Point2f temp;
	if (v1.x >= v2.x) {
		temp = v1;
		v1 = v2;
		v2 = temp;
	}
	float invslope1 = (v3.x - v1.x) / (v3.y - v1.y);
	float invslope2 = (v3.x - v2.x) / (v3.y - v2.y);
	float curx1 = v3.x;
	float curx2 = v3.x;
	for (int scanlineY = (int)v3.y; scanlineY <= v1.y; scanlineY++)
	{
		for (int scanlineX = (int)curx1; scanlineX <= (int)curx2; scanlineX++) {
			points.push_back(Point2f(scanlineX, scanlineY));
		}
		curx1 += invslope1;
		curx2 += invslope2;
	}
}

bool sort_vertices(Point2f &elem1, Point2f &elem2)
{
	return elem1.y >= elem2.y;
}

vector<Point2f> sort_vertices_y_desc(Vec6f t) {
	vector<Point2f> vertices = vector<Point2f>();
	vertices.push_back(Point2f(t[0], t[1]));
	vertices.push_back(Point2f(t[2], t[3]));
	vertices.push_back(Point2f(t[4], t[5]));
	sort(vertices.begin(), vertices.end(), sort_vertices);
	return vertices;
}

vector<Point2f> fill_triangle(Vec6f triangle)
{
	/* at first sort the three vertices by y-coordinate ascending so v1 is the bottom vertice */
	vector<Point2f> sortedVertices = sort_vertices_y_desc(triangle);
	Point2f v1 = sortedVertices[0];
	Point2f v2 = sortedVertices[1];
	Point2f v3 = sortedVertices[2];
	vector<Point2f> points = vector<Point2f>();

	/* here we know that v1.y <= v2.y <= v3.y */
	/* check for trivial case of bottom-flat triangle */
	if (v2.y == v3.y)
	{
		Point2f temp;
		if (v1.x > v2.x) {
			temp = v2;
			v2 = v1;
			v1 = temp;
		}
		fill_bottom_flat_triangle(points, v1, v2, v3);
	}
	/* check for trivial case of top-flat triangle */
	else if (v1.y == v2.y)
	{
		Point2f temp;
		if (v1.x > v2.x) {
			temp = v2;
			v2 = v1;
			v1 = temp;
		}
		fill_top_flat_triangle(points, v1, v2, v3);
	}
	else
	{
		/* general case - split the triangle in a topflat and bottom-flat one */
		Point2f v4 = Point2f(
		(float)(v1.x + ((float)(v2.y - v1.y) / (float)(v3.y - v1.y)) * (v3.x - v1.x)), v2.y);
		fill_bottom_flat_triangle(points, v1, v2, v4);
		fill_top_flat_triangle(points, v2, v4, v3);
	}
	return points;
}

void save_frame_at_tau(Mat &imgA, Mat &imgB, Rect imgRect,
	vector<Mat> affineForward, vector<Mat> affineReverse,
	vector<Vec6f> trianglesA, vector<Vec6f> trianglesB, float tau) {

	int xDim = imgRect.width;
	int yDim = imgRect.height;

	Mat canvas = Mat::zeros(yDim, xDim, CV_8UC1);

	int numTriangles = trianglesA.size();

	std::clock_t start;
	double duration;
	start = clock();

	vector<Vec6f> intTriangles = vector<Vec6f>();
	vector<vector<Point2f>> points = vector<vector<Point2f>>();

	// build up intermediate triangles from T_A, T_B and tau
	for (int i = 0; i < numTriangles; i++) {
		
		float xA = trianglesA[i][0] + (trianglesB[i][0] - trianglesA[i][0])*tau;
		float xB = trianglesA[i][2] + (trianglesB[i][2] - trianglesA[i][2])*tau;
		float xC = trianglesA[i][4] + (trianglesB[i][4] - trianglesA[i][4])*tau;
		float yA = trianglesA[i][1] + (trianglesB[i][1] - trianglesA[i][1])*tau;
		float yB = trianglesA[i][3] + (trianglesB[i][3] - trianglesA[i][3])*tau;
		float yC = trianglesA[i][5] + (trianglesB[i][5] - trianglesA[i][5])*tau;
		intTriangles.push_back(Vec6f(xA, yA, xB, yB, xC, yC));

	}

	// get vector<vector<Point2f>> of points
	for (int i = 0; i < numTriangles; i++) {
		vector<Point2f> triPoints = fill_triangle(intTriangles[i]);
		points.push_back(triPoints);
	}


	// get all transformation params for all triangles

	// combine pixels and weight


	duration = (clock() - start) / (double)CLOCKS_PER_MS;
	cout << duration << endl;

	imshow("purple", canvas);
	waitKey(1);


	cout << "mesh test";

}