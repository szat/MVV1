// mvv_trial.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <ctime>
#include <stdio.h>

using namespace cv;
using namespace std;

static void help()
{
	cout << "\nThis program demostrates iterative construction of\n"
		"delaunay triangulation and voronoi tesselation.\n"
		"It draws a random set of points in an image and then delaunay triangulates them.\n"
		"Usage: \n"
		"./delaunay \n"
		"\nThis program builds the traingulation interactively, you may stop this process by\n"
		"hitting any key.\n";
}

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


void visual_triangulation() {
	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Rect rect(0, 0, 600, 600);

	Subdiv2D subdiv(rect);
	Mat img(rect.size(), CV_8UC3);

	img = Scalar::all(0);
	string win = "Delaunay Demo";
	imshow(win, img);

	std::clock_t start;
	double duration;
	start = std::clock();


	vector<Point2f> points = get_sample_points();
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

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "printf: " << duration << '\n';

	subdiv;

	cin.get();
}

Subdiv2D raw_triangulation() {
	Scalar active_facet_color(0, 0, 255), delaunay_color(255, 255, 255);
	Rect rect(0, 0, 600, 600);

	Subdiv2D subdiv(rect);
	//Mat img(rect.size(), CV_8UC3);

	//img = Scalar::all(0);
	string win = "Delaunay Demo";
	//imshow(win, img);

	std::clock_t start;
	double duration;
	start = std::clock();


	vector<Point2f> points = get_sample_points();
	int numPoints = points.size();

	for (int i = 0; i < numPoints; i++) {
		//locate_point(img, subdiv, points[i], active_facet_color);
		//imshow(win, img);
		//waitKey(1);

		subdiv.insert(points[i]);

		//img = Scalar::all(0);
		//draw_subdiv(img, subdiv, delaunay_color);
		//imshow(win, img);
		//waitKey(1);
	}

	/*
	for (int i = 0; i < 200; i++)
	{
	Point2f fp((float)(rand() % (rect.width - 10) + 5),
	(float)(rand() % (rect.height - 10) + 5));

	locate_point(img, subdiv, fp, active_facet_color);
	imshow(win, img);

	waitKey(1);

	subdiv.insert(fp);

	img = Scalar::all(0);
	draw_subdiv(img, subdiv, delaunay_color);
	imshow(win, img);

	waitKey(1);
	}
	*/

	//img = Scalar::all(0);
	//paint_voronoi(img, subdiv);
	//imshow(win, img);

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "printf: " << duration << '\n';

	return subdiv;
}

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{help h||}");
	if (parser.has("help"))
	{
		help();
		return 0;
	}

	bool graphicalView = false;

	Subdiv2D subdiv;
	if (graphicalView) {
		visual_triangulation();
	}
	else {
		subdiv = raw_triangulation();
	}



	return 0;
}