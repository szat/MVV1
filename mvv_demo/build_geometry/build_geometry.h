#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>

static void draw_subdiv_point(cv::Mat& img, cv::Point2f fp, cv::Scalar color);

static void draw_subdiv(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Scalar delaunay_color);

static void locate_point(cv::Mat& img, cv::Subdiv2D& subdiv, cv::Point2f fp, cv::Scalar active_color);

cv::Subdiv2D graphical_triangulation(std::vector<cv::Point2f> points, cv::Rect source_image_box);

cv::Subdiv2D raw_triangulation(std::vector<cv::Point2f> points, cv::Size size);

void display_triangulation(cv::Subdiv2D subdiv, cv::Rect image_bounds);

std::vector<cv::Vec6f> construct_triangles(std::vector<cv::Point2f> source_image_points, cv::Size source_size);

std::vector<cv::Point2f> convert_key_points(std::vector<cv::KeyPoint> key_points);

std::vector<cv::Vec6f> triangulate_target(std::vector<cv::Point2f> img_points_A, std::vector<cv::Point2f> img_points_B, std::vector<cv::Vec6f> triangles_A);