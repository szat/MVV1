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

void display_triangulation(cv::Subdiv2D subdiv, cv::Rect imageBounds);

std::vector<cv::Vec6f> construct_triangles(std::vector<cv::Point2f> sourceImagePoints, cv::Rect sourceImageBounds);

std::vector<cv::Vec6f> test_interface();

std::vector<cv::Point2f> construct_intermediate_points(std::vector<cv::Point2f> sourcePoints, std::vector<cv::Point2f> targetPoints, int morph);

static void onChangeTriangleMorph(int morph, void *userdata);

int triangulation_trackbar(std::vector<cv::KeyPoint> sourcePoints, std::vector<cv::KeyPoint> targetPoints, cv::Rect imgSize);

std::vector<cv::Point2f> convert_key_points(std::vector<cv::KeyPoint> keyPoints);


