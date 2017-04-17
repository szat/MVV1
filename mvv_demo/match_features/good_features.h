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

static void onChangeTrackbarCorners(int slider, void *userdata);

int trackbarCorners(std::vector<cv::Point2f>& corners);

static void onChange(int trackpos, void *userdata);

int test_trackbar2(int something);
