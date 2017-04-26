#pragma once

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <ctime>
#include <algorithm>
#include <math.h>       /* fabs */

#define M_PI 3.14159265358979323846

bool has_suffix(const std::string &str, const std::string &suffix);

bool has_image_suffix(const std::string &str);

void akaze_script(float akaze_thresh, const cv::Mat& img_in, std::vector<cv::KeyPoint>& kpts_out, cv::Mat& desc_out);

void ratio_matcher_script(const float ratio, const std::vector<cv::KeyPoint>& kpts1_in, const std::vector<cv::KeyPoint>& kpts2_in, const cv::Mat& desc1_in, const cv::Mat& desc2_in, std::vector<cv::KeyPoint>& kpts1_out, std::vector<cv::KeyPoint>& kpts2_out);

void ransac_script(const float ball_radius, const float inlier_thresh, const std::vector<cv::KeyPoint>& kpts1_in, const std::vector<cv::KeyPoint>& kpts2_in, cv::Mat& homography_out, std::vector<cv::KeyPoint>& kpts1_out, std::vector<cv::KeyPoint>& kpts2_out);

void ransac_filtering(float param, const std::vector<cv::KeyPoint> & kptsDomain, const std::vector<cv::KeyPoint> & kptsTarget, const std::vector<int> & indicesDomain, const std::vector<int> & indicesTarget, std::vector<int> indicesDomainPass, std::vector<int> indicesDomainFail, std::vector<int> indicesTargetPass, std::vector<int> indicesTargetFail);

void test_GFTT(std::string imagePathA, std::string imagePathB);

void test_AGAST(std::string imagePathA, std::string imagePathB);

void test_BRISK(std::string imagePathA, std::string imagePathB);

void test_FAST(std::string imagePathA, std::string imagePathB);

void test_ORB(std::string imagePathA, std::string imagePathB);

void affine_skew(double tilt, double phi, cv::Mat& img, cv::Mat& mask, cv::Mat& Ai);

void affine_ORB_detect_and_compute(const cv::Mat& img, std::vector< cv::KeyPoint >& keypoints, cv::Mat& descriptors);

void test_affine_ORB(std::string imagePathA, std::string imagePathB);

void test_kmeans(std::string imagePathA, std::string imagePathB);

std::vector<std::vector<cv::KeyPoint>> match_points_mat(cv::Mat img1, cv::Mat img2);

std::vector<std::vector<cv::KeyPoint>> test_match_points(std::string imagePathA, std::string imagePathB);

std::vector<std::vector<cv::KeyPoint>> test_match_points_2(std::string imagePathA, std::string imagePathB);