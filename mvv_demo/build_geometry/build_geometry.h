// Created by Adrian Szatmari and Daniel Hogg, 2017
// MVV is released under the MIT license
// https://github.com/danielhogg/mvv
// https://github.com/szat/mvv

#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <ctime>

static void draw_subdiv_point(cv::Mat& img, const cv::Point2f & fp, const cv::Scalar & color);

static void draw_subdiv(cv::Mat& img, const cv::Subdiv2D& subdiv, const cv::Scalar & delaunay_color);

static void locate_point(cv::Mat& img, cv::Subdiv2D& subdiv, const cv::Point2f & fp, const cv::Scalar & active_color);

cv::Subdiv2D graphical_triangulation(const std::vector<cv::Point2f> & points, const cv::Rect & source_image_box);

cv::Subdiv2D raw_triangulation(const std::vector<cv::Point2f> & points, const cv::Size & size);

void display_triangulation(const cv::Subdiv2D & subdiv, const cv::Rect & image_bounds);

std::vector<cv::Vec6f> construct_triangles(const std::vector<cv::Point2f> & source_image_points, const cv::Size & source_size);

std::vector<cv::Point2f> convert_key_points(const std::vector<cv::KeyPoint> & key_points);

std::vector<cv::Vec6f> triangulate_target(const std::vector<cv::Point2f> & img1_points, const std::vector<cv::Point2f> & img2_points, const std::vector<cv::Vec6f> & img1_triangles);

long long pair_hash(const cv::Point2f & pt);