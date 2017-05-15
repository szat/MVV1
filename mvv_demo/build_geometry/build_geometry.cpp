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
#include <unordered_map>
#include <limits>
#include <stdlib.h>
#include <math.h>

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

Subdiv2D raw_triangulation(vector<Point2f> points, Size size) {
	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Rect boundingBox = Rect(0, 0, size.width, size.height);
	Subdiv2D subdiv(boundingBox);
	int numPoints = points.size();

	for (int i = 0; i < numPoints; i++) {
		subdiv.insert(points[i]);
	}

	return subdiv;
}

vector<Vec6f> construct_triangles(vector<Point2f> sourceImagePoints, Size sourceSize) {
	// Constructing triangulation of first image

	Subdiv2D subdiv;
	subdiv = raw_triangulation(sourceImagePoints, sourceSize);
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

vector<Point2f> convert_key_points(vector<KeyPoint> keyPoints) {
	int len = keyPoints.size();
	vector<Point2f> result = vector<Point2f>();
	for (int i = 0; i < len; i++) {
		result.push_back(keyPoints[i].pt);
	}
	return result;
}

long long pair_hash(Point2f pt) {
	int imax = std::numeric_limits<int>::max();

	pair<int, int> hash = pair<int, int>();
	long long first = (long long) (pt.x * 1000000);
	long long second = (long long) (pt.y * 1000000);

	return first * (imax + 1) + second;
}

vector<Vec6f> triangulate_target(vector<Point2f> imgPointsA, vector<Point2f> imgPointsB, vector<Vec6f> trianglesA) {
	std::clock_t start;
	double duration;
	start = clock();


	vector<Vec6f> trianglesB = vector<Vec6f>();

	// build up correspondence hashtable
	std::unordered_map<long long, Point2f> pointDict;
	
	int numTriangles = trianglesA.size();
	int numPoints = imgPointsA.size();

	for (int i = 0; i < numPoints; i++) {
		long long hash = pair_hash(imgPointsA[i]);
		pointDict.insert(make_pair(hash, imgPointsB[i]));
	}
	Point2f testPoint = pointDict[pair_hash(imgPointsA[0])];

	duration = (clock() - start) / (double)CLOCKS_PER_MS;
	cout << "built dictionary in " << duration << " ms" << endl;

	for (int i = 0; i < numTriangles; i++) {
		Vec6f currentTriangleA = trianglesA[i];
		Point2f vertex1 = Point2f(currentTriangleA[0], currentTriangleA[1]);
		Point2f vertex2 = Point2f(currentTriangleA[2], currentTriangleA[3]);
		Point2f vertex3 = Point2f(currentTriangleA[4], currentTriangleA[5]);
	
		Point2f newVertex1 = pointDict[pair_hash(vertex1)];
		Point2f newVertex2 = pointDict[pair_hash(vertex2)];
		Point2f newVertex3 = pointDict[pair_hash(vertex3)];

		Vec6f triangleB = Vec6f(newVertex1.x, newVertex1.y, newVertex2.x, newVertex2.y, newVertex3.x, newVertex3.y);
		trianglesB.push_back(triangleB);
	}

	return trianglesB;
}

