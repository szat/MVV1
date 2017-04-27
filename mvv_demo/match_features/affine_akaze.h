#pragma once

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>

void affine_skew_here(double tilt, double phi, cv::Mat& img, cv::Mat& mask, cv::Mat& Ai);

void detect_and_compute_here(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

void affine_akaze_test(std::string imagePathA_in, std::string imagePathB_in, std::vector<cv::KeyPoint>& keysImgA_out, std::vector<cv::KeyPoint>& keysImgB_out);
