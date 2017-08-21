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
	vector<Vec6f> triangle_list;
	subdiv.getTriangleList(triangle_list);
	vector<Point> pt(3);

	int num_triangles = triangle_list.size();

	for (size_t i = 0; i < num_triangles; i++)
	{
		Vec6f t = triangle_list[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[1], pt[2], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[2], pt[0], delaunay_color, 1, LINE_AA, 0);
	}

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

Subdiv2D graphical_triangulation(vector<Point2f> points, Rect source_image_box) {
	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);

	Subdiv2D subdiv(source_image_box);
	Mat img(source_image_box.size(), CV_8UC3);

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
	Rect bounding_box = Rect(0, 0, size.width, size.height);
	Subdiv2D subdiv(bounding_box);
	int numPoints = points.size();

	for (int i = 0; i < numPoints; i++) {
		subdiv.insert(points[i]);
	}

	return subdiv;
}

vector<Vec6f> construct_triangles(vector<Point2f> source_image_points, Size source_size) {
	// Constructing triangulation of first image

	Subdiv2D subdiv;
	subdiv = raw_triangulation(source_image_points, source_size);
	vector<Vec6f> triangles = vector<Vec6f>();
	subdiv.getTriangleList(triangles);

	int num_exterior_points = 4;
	vector<Point2f> excluded_vertices = vector<Point2f>();
	for (int i = 0; i < num_exterior_points; i++) {
		excluded_vertices.push_back(subdiv.getVertex(i));
	}

	vector<Vec6f> filtered_triangles = vector<Vec6f>();
	// Is there an alternative to a 3-deep nested loop?

	int num_triangles = triangles.size();
	for (int i = 0; i < num_triangles; i++) {
		bool exclude = false;
		for (int j = 0; j < num_exterior_points; j++) {
			Point2f excluded_vertex = excluded_vertices[j];
			for (int k = 0; k < 3; k++) {
				float x_0 = triangles[i][k * 2];
				float y_0 = triangles[i][k * 2 + 1];
				if (x_0 == excluded_vertex.x && y_0 == excluded_vertex.y) {
					exclude = true;
				}
			}
		}
		if (!exclude) {
			filtered_triangles.push_back(triangles[i]);
		}
	}
	return filtered_triangles;
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

vector<Point2f> convert_key_points(vector<KeyPoint> key_points) {
	int len = key_points.size();
	vector<Point2f> result = vector<Point2f>();
	for (int i = 0; i < len; i++) {
		result.push_back(key_points[i].pt);
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

vector<Vec6f> triangulate_target(vector<Point2f> img_points_A, vector<Point2f> img_points_B, vector<Vec6f> triangles_A) {
	std::clock_t start;
	double duration;
	start = clock();


	vector<Vec6f> triangles_B = vector<Vec6f>();

	// build up correspondence hashtable
	std::unordered_map<long long, Point2f> point_dict;
	
	int numTriangles = triangles_A.size();
	int numPoints = img_points_A.size();

	for (int i = 0; i < numPoints; i++) {
		long long hash = pair_hash(img_points_A[i]);
		point_dict.insert(make_pair(hash, img_points_B[i]));
	}
	Point2f test_point = point_dict[pair_hash(img_points_A[0])];

	duration = (clock() - start) / (double)CLOCKS_PER_MS;
	cout << "built dictionary in " << duration << " ms" << endl;

	for (int i = 0; i < numTriangles; i++) {
		Vec6f current_triangle_A = triangles_A[i];
		Point2f vertex1 = Point2f(current_triangle_A[0], current_triangle_A[1]);
		Point2f vertex2 = Point2f(current_triangle_A[2], current_triangle_A[3]);
		Point2f vertex3 = Point2f(current_triangle_A[4], current_triangle_A[5]);
	
		Point2f new_vertex_1 = point_dict[pair_hash(vertex1)];
		Point2f new_vertex_2 = point_dict[pair_hash(vertex2)];
		Point2f new_vertex_3 = point_dict[pair_hash(vertex3)];

		Vec6f triangle_B = Vec6f(new_vertex_1.x, new_vertex_1.y, new_vertex_2.x, new_vertex_2.y, new_vertex_3.x, new_vertex_3.y);
		triangles_B.push_back(triangle_B);
	}

	return triangles_B;
}