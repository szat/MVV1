#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

cv::Mat merge_images(cv::Mat img_1, cv::Mat img_2);
