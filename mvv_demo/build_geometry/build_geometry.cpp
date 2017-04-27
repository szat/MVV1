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
	long long first = (long long) (pt.x * 1000);
	long long second = (long long) (pt.y * 1000);

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

void render_triangles(vector<Vec6f> triangles, Rect bounds) {

	cout << "Rendering bounds";
}

vector<int> get_source_convex_hull(vector<Point2f> sourcePoints) {
	vector<int> hull = vector<int>();
	int numPoints = sourcePoints.size();
	vector<Point> sourcePointsSimple = vector<Point>();
	for (int i = 0; i < numPoints; i++) {
		Point2f currentPoint = sourcePoints[i];
		sourcePointsSimple.push_back(Point(currentPoint.x, currentPoint.y));
	}

	convexHull(sourcePointsSimple, hull, true);

	return hull;
}






vector<Point2f> hull_indices_to_points(vector<int> indices, vector<Point2f> points) {
	vector<Point2f> resultPoints = vector<Point2f>();
	int numIndices = indices.size();

	for (int i = 0; i < numIndices; i++) {
		int currentIndex = indices[i];
		resultPoints.push_back(points[currentIndex]);
	}
	return resultPoints;
}

Point2f get_center_of_mass(vector<Point2f> points) {
	// Using 2D center of mass formula with equal masses (M=1)
	int numPoints = points.size();
	float reciprocal = 1 / ((float)numPoints);
	float xTotal = 0;
	float yTotal = 0;

	for (int i = 0; i < numPoints; i++) {
		xTotal += points[i].x;
		yTotal += points[i].y;
	}

	float xMass = reciprocal * xTotal;
	float yMass = reciprocal * yTotal;

	return Point2f(xMass, yMass);
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (p1, q1) and (p2, q2).
bool intersection(Point2f p1, Point2f q1, Point2f p2, Point2f q2,
	Point2f &r)
{
	Point2f x = p2 - p1;
	Point2f d1 = q1 - p1;
	Point2f d2 = q2 - p2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = p1 + d1 * t1;
	return true;
}

bool validate_edge_point(Point2f edgePoint, Point2f hullPoint, Point2f com, float xMin, float xMax, float yMin, float yMax) {
	// With a tolerance this large, I think this will eventually cause problems via random chance
	// Eventually, we will get a point that gets sent to the corner, and matches both lines.

	// WAIT! We can fix this by (if we have more than two), checking which one has a lower tolerance to the zero line.

	float tolerance = 1e-4;
	if (edgePoint.x + tolerance < xMin || edgePoint.x - tolerance > xMax) {
		return false;
	}
	if (edgePoint.y + tolerance < yMin || edgePoint.y - tolerance > yMax) {
		return false;
	}

	float xDiff = com.x - hullPoint.x;
	float yDiff = com.y - hullPoint.y;
	float xEdgeDiff = com.x - edgePoint.x;
	float yEdgeDiff = com.y - edgePoint.y;
	bool xDiffSign = xDiff < 0 ? false : true;
	bool yDiffSign = yDiff < 0 ? false : true;
	bool xEdgeDiffSign = xEdgeDiff < 0 ? false : true;
	bool yEdgeDiffSign = yEdgeDiff < 0 ? false : true;

	if (xDiffSign == xEdgeDiffSign && yDiffSign == yEdgeDiffSign) {
		return true;
	}
	else {
		return false;
	}
}

Point2f find_edge_intersect(Point2f hullPoint, Point2f com, Rect imgBounds) {

	float xMin = 0;
	float xMax = imgBounds.size().width;
	float yMin = 0;
	float yMax = imgBounds.size().height;

	// 4 lines form the box
	// what a useless comment
	Point2f pointA, pointB, pointC, pointD;
	bool pointIntersectA, pointIntersectB, pointIntersectC, pointIntersectD;

	pointIntersectA = intersection(com, hullPoint, Point2f(xMin, yMin), Point2f(xMin, yMax), pointA);
	pointIntersectB = intersection(com, hullPoint, Point2f(xMin, yMax), Point2f(xMax, yMax), pointB);
	pointIntersectC = intersection(com, hullPoint, Point2f(xMax, yMax), Point2f(xMax, yMin), pointC);
	pointIntersectD = intersection(com, hullPoint, Point2f(xMax, yMin), Point2f(xMin, yMin), pointD);

	vector<Point2f> validEdgePoints = vector<Point2f>();

	if (pointIntersectA && validate_edge_point(pointA, hullPoint, com, xMin, xMax, yMin, yMax)) {
		validEdgePoints.push_back(pointA);
	}
	if (pointIntersectB && validate_edge_point(pointB, hullPoint, com, xMin, xMax, yMin, yMax)) {
		validEdgePoints.push_back(pointB);
	}
	if (pointIntersectC && validate_edge_point(pointC, hullPoint, com, xMin, xMax, yMin, yMax)) {
		validEdgePoints.push_back(pointC);
	}
	if (pointIntersectD && validate_edge_point(pointD, hullPoint, com, xMin, xMax, yMin, yMax)) {
		validEdgePoints.push_back(pointD);
	}

	if (validEdgePoints.size() != 1) {
		throw "Error! Edge intersects != 1!";
	}
	else {
		return validEdgePoints[0];
	}
}

vector<pair<Vec4f, Vec4f>> project_trapezoids_from_hull(vector<Point2f> convexHull, Rect imgBounds, Point2f centerOfMass) {
	vector<Point2f> exteriorProjection = vector<Point2f>();
	vector<pair<Vec4f, Vec4f>> trapezoids = vector<pair<Vec4f,Vec4f>>();
	int hullPoints = convexHull.size();
	for (int i = 0; i < hullPoints; i++) {
		Point2f poi = find_edge_intersect(convexHull[i], centerOfMass, imgBounds);
		exteriorProjection.push_back(poi);
	}

	for (int i = 0; i < hullPoints; i++) {
		int firstIndex = i;
		int secondIndex = (i + 1) % hullPoints;

		Vec4f xCoords = Vec4f(convexHull[firstIndex].x, exteriorProjection[firstIndex].x, exteriorProjection[secondIndex].x, convexHull[secondIndex].x);
		Vec4f yCoords = Vec4f(convexHull[firstIndex].y, exteriorProjection[firstIndex].y, exteriorProjection[secondIndex].y, convexHull[secondIndex].y);
		pair<Vec4f, Vec4f> trapezoid = pair<Vec4f, Vec4f>(xCoords, yCoords);
		trapezoids.push_back(trapezoid);
	}
	return trapezoids;
};

float get_triangle_area(Vec6f t) {
	// Heron's formula

	float a2 = (t[2] - t[0])*(t[2] - t[0]) + (t[3] - t[1])*(t[3] - t[1]);
	float b2 = (t[4] - t[2])*(t[4] - t[2]) + (t[5] - t[3])*(t[5] - t[3]);
	float c2 = (t[0] - t[4])*(t[0] - t[4]) + (t[1] - t[5])*(t[1] - t[5]);

	return (0.25)*sqrt(4 * a2*b2 - (a2 + b2 - c2)*(a2 + b2 - c2));
}

float get_trapezoid_area(pair<Vec4f,Vec4f> t) {

	// make two triangles, add them together.
	Vec6f t1 = Vec6f(t.first[0], t.second[0], t.first[1], t.second[1], t.first[2], t.second[2]);
	Vec6f t2 = Vec6f(t.first[0], t.second[0], t.first[2], t.second[2], t.first[3], t.second[3]);

	float area1 = get_triangle_area(t1);
	float area2 = get_triangle_area(t2);

	return area1 + area2;
}

struct polygon_sort
{
	inline bool operator() (const pair<int, float>& pair1, const pair<int, float>& pair2) {
		return (pair1.second < pair2.second);
	}
};

vector<int> calculate_triangle_priority(vector<Vec6f> triangles) {
	// Lower numbers indicate higher priority
	// Larger triangles have lower priority

	int len = triangles.size();
	vector<pair<int,float>> triangleArea = vector<pair<int,float>>();
	for (int i = 0; i < len; i++) {
		pair<int, float> currentT = pair<int, float>();
		currentT.first = i;
		currentT.second = get_triangle_area(triangles[i]);
		triangleArea.push_back(currentT);
	}
	sort(triangleArea.begin(), triangleArea.end(), polygon_sort());
	vector<int> priority = vector<int>();
	for (int i = 0; i < len; i++) {
		priority.push_back(triangleArea[i].first);
	}
	return priority;
}

vector<int> calculate_trapezoid_priority(vector<pair<Vec4f, Vec4f>> trapezoids) {
	// Lower numbers indicate higher priority
	// Larger trapezoids have lower priority

	int len = trapezoids.size();
	vector<pair<int, float>> trapezoidArea = vector<pair<int, float>>();
	for (int i = 0; i < len; i++) {
		pair<int, float> currentT = pair<int, float>();
		currentT.first = i;
		currentT.second = get_trapezoid_area(trapezoids[i]);
		trapezoidArea.push_back(currentT);
	}
	sort(trapezoidArea.begin(), trapezoidArea.end(), polygon_sort());
	vector<int> priority = vector<int>();
	for (int i = 0; i < len; i++) {
		priority.push_back(trapezoidArea[i].first);
	}
	return priority;
}