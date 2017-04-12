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
#include "stdafx.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>
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


vector<Point2f> get_one_sample_point() {
	vector<Point2f> points = vector<Point2f>();
	points.push_back(Point2f(500, 20));
	return points;
}

vector<Point2f> get_small_sample_points() {
	vector<Point2f> points = vector<Point2f>();
	points.push_back(Point2f(528, 390));
	points.push_back(Point2f(551, 209));
	points.push_back(Point2f(289, 355));
	points.push_back(Point2f(569, 105));
	points.push_back(Point2f(155, 551));
	return points;
}

vector<Point2f> get_sample_points() {
	vector<Point2f> points = vector<Point2f>();
	points.push_back(Point2f(528, 390));
	points.push_back(Point2f(551, 209));
	points.push_back(Point2f(289, 355));
	points.push_back(Point2f(569, 105));
	points.push_back(Point2f(155, 551));
	points.push_back(Point2f(335, 319));
	points.push_back(Point2f(273, 331));
	points.push_back(Point2f(353, 505));
	points.push_back(Point2f(590, 287));
	points.push_back(Point2f(305, 31));
	points.push_back(Point2f(25, 313));
	points.push_back(Point2f(16, 374));
	points.push_back(Point2f(44, 57));
	points.push_back(Point2f(517, 335));
	points.push_back(Point2f(564, 475));
	points.push_back(Point2f(508, 548));
	points.push_back(Point2f(126, 494));
	points.push_back(Point2f(553, 267));
	points.push_back(Point2f(52, 565));
	points.push_back(Point2f(91, 589));
	points.push_back(Point2f(234, 478));
	points.push_back(Point2f(179, 211));
	points.push_back(Point2f(261, 324));
	points.push_back(Point2f(392, 91));
	points.push_back(Point2f(560, 592));
	points.push_back(Point2f(578, 383));
	points.push_back(Point2f(260, 316));
	points.push_back(Point2f(529, 14));
	points.push_back(Point2f(70, 507));
	points.push_back(Point2f(264, 477));
	points.push_back(Point2f(188, 119));
	points.push_back(Point2f(156, 125));
	points.push_back(Point2f(207, 251));
	points.push_back(Point2f(162, 214));
	points.push_back(Point2f(218, 592));
	points.push_back(Point2f(504, 38));
	points.push_back(Point2f(319, 267));
	points.push_back(Point2f(21, 31));
	points.push_back(Point2f(106, 88));
	points.push_back(Point2f(508, 158));
	points.push_back(Point2f(482, 158));
	points.push_back(Point2f(419, 335));
	points.push_back(Point2f(332, 554));
	points.push_back(Point2f(538, 589));
	points.push_back(Point2f(133, 198));
	points.push_back(Point2f(160, 9));
	points.push_back(Point2f(436, 462));
	points.push_back(Point2f(256, 442));
	points.push_back(Point2f(159, 130));
	points.push_back(Point2f(97, 150));
	return points;
}


Subdiv2D graphical_triangulation(vector<Point2f> points) {
	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Rect rect(0, 0, 600, 600);

	Subdiv2D subdiv(rect);
	Mat img(rect.size(), CV_8UC3);

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

Subdiv2D raw_triangulation(vector<Point2f> points) {
	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Rect rect(0, 0, 600, 600);
	Subdiv2D subdiv(rect);
	int numPoints = points.size();

	for (int i = 0; i < numPoints; i++) {
		subdiv.insert(points[i]);
	}

	return subdiv;
}

class MatchedFeature {
public:
	MatchedFeature(float sourceX, float sourceY, float targetX, float targetY) {
		srcX = sourceX;
		srcY = sourceY;
		tarX = targetX;
		tarY = targetY;
		
	}
protected:
	float srcX;
	float srcY;
	float tarX;
	float tarY;

	/*
	Need getters for the matched features. 
	No setters though!! The values don't change once initialized.
	*/

};


void construct_geometries(vector<MatchedFeature> matchedFeatures, Rect sourceImageBounds, Rect destImageBounds) {
	cout << "Not presently implemented";
}

int main(int argc, char** argv)
{
	// Secure input arguments in main
	// Program parameters:
	// vector<Point2f> points
	// Rect (cv type) boundingBox
	// bool indicating graphical or not


	bool graphics = true;
	string pointSample = "m";

	vector<Point2f> points;
	if (pointSample == "s") {
		points = get_small_sample_points();
	}
	else if (pointSample == "m") {
		points = get_sample_points();
	}
	else if (pointSample == "o") {
		points = get_one_sample_point();
	}
	else {
		cout << "Exception: No point sample matches this input.\n";
		cin.get();
		return -1;
	}

	Subdiv2D subdiv;
	string processMessage;

	// timing triangulation process
	std::clock_t start;
	double duration;
	start = clock();

	if (graphics) {
		processMessage = "Graphical triangulation time: ";
		subdiv = graphical_triangulation(points);
	}
	else {
		processMessage = "Raw triangulation time: ";
		subdiv = raw_triangulation(points);
	}
	// duration of raw or visual triangulation.
	duration = (clock() - start) / (double)CLOCKS_PER_MS;
	cout << processMessage << duration << "ms" << "\n" ;

	cout << "Completed triangulation on source image.\n";
	cout << "Program will now propagate triangulation to target image.\n";

	// maybe modify the subdiv to eliminate edges and vertrtices before making trianges???

	vector<Vec6f> triangles = vector<Vec6f>();
	subdiv.getTriangleList(triangles);

	// eliminate unecessary vertices (no, above step is better)

	cin.get();

	return 0;
}