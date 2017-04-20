#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <iostream>
#include <ctime>

void changeCornersMaxCorners(int maxCornersSlider, void *userdata);

void changeCornersBlockSize(int blockSizeSlider, void *userdata);

void changeCornersQualityLevel(int qualityLevelInt, void *userdata);

void changeCornersMinDistance(int minDistanceInt, void *userdata);

//Useless method unless useHarrisDetector == true
void changeCornersKInt(int kInt, void *userdata);

int trackbarCorners(std::vector<cv::Point2f>& corners);