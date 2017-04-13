#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <stdio.h>

static void draw_subdiv_point(cv::Mat& img, cv::Point2f fp, cv::Scalar color);

static void draw_subdiv(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color);

static void locate_point(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Point2f fp, cv::Scalar active_color);

cv::Subdiv2D graphical_triangulation(std::vector<cv::Point2f> points, cv::Rect sourceImageBoundingBox);

cv::Subdiv2D raw_triangulation(std::vector<cv::Point2f> points, cv::Rect sourceImageBoundingBox);

std::vector<cv::Vec6f> construct_triangles(std::vector<cv::Point2f> sourceImagePoints, cv::Rect sourceImageBounds);

std::vector<cv::Vec6f> test_interface();




