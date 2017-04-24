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

std::vector<cv::Vec6f> triangulate_target(std::vector<cv::Point2f> imgPointsA, std::vector<cv::Point2f> imgPointsB, std::vector<cv::Vec6f> trianglesA);

void render_triangles(std::vector<cv::Vec6f> triangles, cv::Rect bounds);

std::vector<int> get_source_convex_hull(std::vector<cv::Point2f> sourcePoints);

std::vector<cv::Point2f> hull_indices_to_points(std::vector<int> indices, std::vector<cv::Point2f> points);

std::vector<std::pair<cv::Vec4f, cv::Vec4f>> project_trapezoids_from_hull(std::vector<cv::Point2f> convexHull, cv::Rect imgBounds, cv::Point2f centerOfMass);

bool intersection(cv::Point2f p1, cv::Point2f q1, cv::Point2f p2, cv::Point2f q2, cv::Point2f &r);

bool validate_edge_point(cv::Point2f edgePoint, cv::Point2f hullPoint, cv::Point2f com);

cv::Point2f find_edge_intersect(cv::Point2f hullPoint, cv::Point2f com, cv::Rect imgBounds);

cv::Point2f get_center_of_mass(std::vector<cv::Point2f> points);

std::vector<int> calculate_triangle_priority(std::vector<cv::Vec6f> triangles);

std::vector<int> calculate_trapezoid_priority(std::vector<std::pair<cv::Vec4f, cv::Vec4f>> trapezoids);