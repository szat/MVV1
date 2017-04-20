/*
This code was adapted from the Delaunay demo code from openCV, and so we have attached
the following BSD license.

-----------------------------------------------

By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.

License Agreement
For Open Source Computer Vision Library

(3 - clause BSD License)

Copyright(C) 2000 - 2016, Intel Corporation, all rights reserved.
Copyright(C) 2009 - 2011, Willow Garage Inc., all rights reserved.
Copyright(C) 2009 - 2016, NVIDIA Corporation, all rights reserved.
Copyright(C) 2010 - 2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright(C) 2015 - 2016, OpenCV Foundation, all rights reserved.
Copyright(C) 2015 - 2016, Itseez Inc., all rights reserved.

Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met :

* Redistributions of source code must retain the above copyright notice,

this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,

this list of conditions and the following disclaimer in the documentation
and / or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.

In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort(including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>

#include "build_geometry.h"
#include "generate_test_points.h"

#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)

using namespace cv;
using namespace std;

static void draw_subdiv_point(Mat& img, Point2f fp, Scalar color)
{
	circle(img, fp, 3, color, FILLED, LINE_8, 0);
}

static void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
#if 1
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	int numTriangles = triangleList.size();

	for (size_t i = 0; i < numTriangles; i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[1], pt[2], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[2], pt[0], delaunay_color, 1, LINE_AA, 0);
	}
#else
	vector<Vec4f> edgeList;
	subdiv.getEdgeList(edgeList);
	for (size_t i = 0; i < edgeList.size(); i++)
	{
		Vec4f e = edgeList[i];
		Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
		Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
		line(img, pt0, pt1, delaunay_color, 1, LINE_AA, 0);
	}
#endif
}

static void locate_point(Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color)
{
	int e0 = 0, vertex = 0;

	subdiv.locate(fp, e0, vertex);

	if (e0 > 0)
	{
		int e = e0;
		do
		{
			Point2f org, dst;
			if (subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0)
				line(img, org, dst, active_color, 3, LINE_AA, 0);

			e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_LEFT);
		} while (e != e0);
	}

	draw_subdiv_point(img, fp, active_color);
}

Subdiv2D graphical_triangulation(vector<Point2f> points, Rect sourceImageBoundingBox) {
	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);

	Subdiv2D subdiv(sourceImageBoundingBox);
	Mat img(sourceImageBoundingBox.size(), CV_8UC3);

	img = Scalar::all(0);
	string win = "Delaunay Demo";
	imshow(win, img);

	int numPoints = points.size();

	for (int i = 0; i < numPoints; i++) {
		locate_point(img, subdiv, points[i], active_facet_color);
		imshow(win, img);
		waitKey(1);

		subdiv.insert(points[i]);

		img = Scalar::all(0);
		draw_subdiv(img, subdiv, delaunay_color);
		imshow(win, img);
		waitKey(1);
	}

	img = Scalar::all(0);
	return subdiv;
}

Subdiv2D raw_triangulation(vector<Point2f> points, Rect sourceImageBoundingBox) {
	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Subdiv2D subdiv(sourceImageBoundingBox);
	int numPoints = points.size();

	for (int i = 0; i < numPoints; i++) {
		subdiv.insert(points[i]);
	}

	return subdiv;
}

vector<Vec6f> construct_triangles(vector<Point2f> sourceImagePoints, Rect sourceImageBounds) {
	// Constructing triangulation of first image

	Subdiv2D subdiv;
	subdiv = raw_triangulation(sourceImagePoints, sourceImageBounds);
	vector<Vec6f> triangles = vector<Vec6f>();
	subdiv.getTriangleList(triangles);

	int numExteriorPoints = 4;
	vector<Point2f> excludedVertices = vector<Point2f>();
	for (int i = 0; i < numExteriorPoints; i++) {
		excludedVertices.push_back(subdiv.getVertex(i));
	}

	vector<Vec6f> filteredTriangles = vector<Vec6f>();
	// Is there an alternative to a 3-deep nested loop?

	int numTriangles = triangles.size();
	for (int i = 0; i < numTriangles; i++) {
		bool exclude = false;
		for (int j = 0; j < numExteriorPoints; j++) {
			Point2f excludedVertex = excludedVertices[j];
			for (int k = 0; k < 3; k++) {
				float x_0 = triangles[i][k * 2];
				float y_0 = triangles[i][k * 2 + 1];
				if (x_0 == excludedVertex.x && y_0 == excludedVertex.y) {
					exclude = true;
				}
			}
		}
		if (!exclude) {
			filteredTriangles.push_back(triangles[i]);
		}
	}
	return filteredTriangles;
}

void display_triangulation(Subdiv2D subdiv, Rect imageBounds) {
	// the graphical_triangulation function is far too slow

	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Mat img(imageBounds.size(), CV_8UC3);

	img = Scalar::all(0);
	string win = "Delaunay Demo";
	imshow(win, img);

	draw_subdiv(img, subdiv, delaunay_color);
	imshow(win, img);
	waitKey(1);
}

vector<Vec6f> test_interface()
{
	string input = "";
	int numberPoints = 0;

	while (true) {
		cout << "Please enter the number of feature points you wish to test: ";
		getline(cin, input);

		stringstream myStream(input);
		if (myStream >> numberPoints)
			break;
		cout << "Invalid number, please try again." << endl;
	}

	char graphics = { 0 };
	bool graphicsMode = false;

	while (true) {
		cout << "Graphical mode? (y/n)" << endl;
		getline(cin, input);

		if (input.length() == 1) {
			graphics = input[0];
			if (graphics == 'y' || graphics == 'Y') {
				graphicsMode = true;
				break;
			}
			else if (graphics == 'n' || graphics == 'N') {
				graphicsMode = false;
				break;
			}
		}

		cout << "Invalid character, please try again." << endl;
	}

	cout << "Running delaunay triangulation on " << numberPoints << " vertices." << endl;

	Rect rect(0, 0, 600, 600);
	vector<Point2f> points = get_n_random_points(rect, numberPoints);

	Subdiv2D subdiv;
	string processMessage;

	// timing triangulation process
	std::clock_t start;
	double duration;
	start = clock();
	subdiv = raw_triangulation(points, rect);

	if (graphicsMode) {
		display_triangulation(subdiv, rect);
		processMessage = "Graphical triangulation time: ";
	}
	else {
		processMessage = "Raw triangulation time: ";
	}
	// duration of raw or visual triangulation.
	duration = (clock() - start) / (double)CLOCKS_PER_MS;
	cout << processMessage << duration << "ms" << "\n";

	cout << "Completed triangulation on source image.\n";

	// maybe modify the subdiv to eliminate edges and vertrtices before making trianges???

	vector<Vec6f> triangles = vector<Vec6f>();
	subdiv.getTriangleList(triangles);
	return triangles;
}

struct trackbarTriangleMorph {
	Rect imageSize;
	int morph;
	vector<KeyPoint> sourcePoints;
	vector<KeyPoint> targetPoints;
};

vector<Point2f> construct_intermediate_points(vector<KeyPoint> sourcePoints, vector<KeyPoint> targetPoints, int morph) {
	float morphFactor = (float)morph / 100;
	vector<Point2f> intermediate = vector<Point2f>();

	int numPoints = sourcePoints.size();

	for (int i = 0; i < numPoints; i++) {
		Point2f srcPoint = sourcePoints[i].pt;
		Point2f tarPoint = targetPoints[i].pt;
		float x_0 = srcPoint.x;
		float x_1 = tarPoint.x;
		float y_0 = srcPoint.y;
		float y_1 = tarPoint.y;

		float x_diff = x_1 - x_0;
		float y_diff = y_1 - y_0;

		float x_morph = x_0 + x_diff * morphFactor;
		float y_morph = y_0 + y_diff * morphFactor;

		Point2f intermediatePoint = Point2f(x_morph, y_morph);
		intermediate.push_back(intermediatePoint);
	}
	return intermediate;
}

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

int triangulation_trackbar(vector<KeyPoint> sourcePoints, vector<KeyPoint> targetPoints, Rect imgSize)
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

vector<Point2f> convert_key_points(vector<KeyPoint> keyPoints) {
	int len = keyPoints.size();
	vector<Point2f> result = vector<Point2f>();
	for (int i = 0; i < len; i++) {
		result.push_back(keyPoints[i].pt);
	}
	return result;
}