#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>

static void draw_subdiv_point(Mat& img, Point2f fp, Scalar color);

static void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color);

static void locate_point(Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color);

Subdiv2D graphical_triangulation(vector<Point2f> points, Rect sourceImageBoundingBox);

Subdiv2D raw_triangulation(vector<Point2f> points, Rect sourceImageBoundingBox);

vector<Vec6f> construct_triangles(vector<Point2f> sourceImagePoints, Rect sourceImageBounds);

vector<Vec6f> test_interface();


