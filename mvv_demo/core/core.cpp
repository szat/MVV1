// core.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <vector>
#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>

#include "binary_write.h"
#include "merge_background.h"
#include "video_preprocessing.h"

#include "build_geometry.h"
#include "generate_test_points.h"

#include "match_points.h"
#include "good_features.h"
#include "affine_akaze.h"
#include "knn_test.h"
#include "background_substraction.h"

#include "interpolate_images.h"
#include "polygon_raster.h"


#define VERSION "1.0.0"
#define APPLICATION_NAME "MVV"
#define COMPANY_NAME "NDim Inc."
#define COPYRIGHT_YEAR 2017
#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)
#define NUM_THREADS 16

typedef std::chrono::high_resolution_clock Clock;

using namespace std;
using namespace cv;

int corner_points_test() {
	cout << "Begin integration test of match_features and build_geometry" << endl;

	//this is the image used in trackbarCorners
	string image_path = "..\\data_store\\david_1.jpg";
	Mat src1 = imread(image_path, IMREAD_GRAYSCALE); 
	vector<Point2f> corners;
	trackbar_corners(image_path, corners);


	Rect test_rect = Rect(0, 0, src1.size().width, src1.size().height);
	graphical_triangulation(corners, test_rect);

	return 0;
}

Rect get_image_size(string imagePath) {
	string address = "..\\data_store\\" + imagePath;
	string input = "";
	ifstream infile1;
	infile1.open(address.c_str());

	Mat img1 = imread(address, IMREAD_GRAYSCALE);

	float height = img1.size().height;
	float width = img1.size().width;

	Rect image_size = Rect(0, 0, width, height);
	return image_size;
}

int test_matching() {

	string imageA = "david_1.jpg";
	string imageB = "david_2.jpg";

	Rect image_size_A = get_image_size(imageA);
	Rect image_size_B = get_image_size(imageB);

	vector<vector<KeyPoint>> point_matches = test_match_points(imageA, imageB);

	vector<KeyPoint> image_matchesA = point_matches.at(0);
	vector<KeyPoint> image_matchesB = point_matches.at(1);

	vector<Point2f> image_pointsA = vector<Point2f>();

	int vecSize = image_matchesA.size();

	for (int i = 0; i < vecSize; i++) {
		image_pointsA.push_back(image_matchesA.at(i).pt);
	}

	//graphical_triangulation(imagePointsA, imageSizeA);
	//Subdiv2D subdiv = raw_triangulation(imagePointsA, imageSizeA);
	//display_triangulation(subdiv, imageSizeA);

	return 0;
}

struct GeometricSlice {
	Rect img;
	vector<Vec6f> triangles;
};

struct MatchedGeometry {
	GeometricSlice source_geometry;
	GeometricSlice target_geometry;
};

MatchedGeometry create_matched_geometry(vector<Point2f> imgA_points, vector<Point2f> imgB_points, Size size) {
	// triangulate source interior
	vector<Vec6f> trianglesA = construct_triangles(imgA_points, size);

	// triangulate target interior
	vector<Vec6f> trianglesB = triangulate_target(imgA_points, imgB_points, trianglesA);

	Rect img_bounds = Rect(0, 0, size.width, size.height);

	// This could potentially be replaced by two constructors.
	MatchedGeometry matched_result = MatchedGeometry();
	GeometricSlice source = GeometricSlice();
	GeometricSlice target = GeometricSlice();
	source.triangles = trianglesA;
	target.triangles = trianglesB;
	source.img = img_bounds;
	target.img = img_bounds;
	matched_result.source_geometry = source;
	matched_result.target_geometry = target;
	return matched_result;
}

MatchedGeometry read_matched_points_from_file(Mat &img1, Mat &img2, Size video_size) {
	// Please note that:
	// A: source image
	// B: target image
	// I use this shorthand so that the variable names are shorter.

	cout << "Initializing matched geometry routine" << endl;

	Mat imgA;
	Mat imgB;
	cvtColor(img1, imgA, CV_BGR2GRAY);
	cvtColor(img2, imgB, CV_BGR2GRAY);

	// here we are assuming the images are the same size
	//resize(imgA, imgA, desired_size);

	vector<vector<KeyPoint>> point_matches = match_points_mat(imgA, imgB);

	vector<KeyPoint> imgA_keypoints = point_matches[0];
	vector<KeyPoint> imgB_keypoints = point_matches[1];
	vector<Point2f> imgA_points = convert_key_points(imgA_keypoints);
	vector<Point2f> imgB_points = convert_key_points(imgB_keypoints);

	// Rescaling points
	/*
	float x_scaling = (float)desired_size.width / (float)original_size.width;
	float y_scaling = (float)desired_size.height / (float)original_size.height;

	vector<Point2f> imgA_points_rescaled = vector<Point2f>();
	vector<Point2f> imgB_points_rescaled = vector<Point2f>();

	int num_points = min(imgA_points.size(), imgB_points.size());

	for (int i = 0; i < num_points; i++) {
		float new_x_A = x_scaling * imgA_points[i].x;
		float new_y_A = y_scaling * imgA_points[i].y;
		Point2f pointA = Point2f(new_x_A, new_y_A);
		float new_x_B = x_scaling * imgB_points[i].x;
		float new_y_B = y_scaling * imgB_points[i].y;
		Point2f pointB = Point2f(new_x_B, new_y_B);
		imgA_points_rescaled.push_back(pointA);
		imgB_points_rescaled.push_back(pointB);
	}
	*/
	MatchedGeometry geometry = create_matched_geometry(imgA_points, imgB_points, video_size);
	return geometry;
}

void render_matched_geometry(GeometricSlice slice, string window_name) {
	// Render both images.

	// Trapezoids in red (first)
	Mat img(slice.img.size(), CV_8UC3);
	Scalar trapezoid_color(0, 0, 255), triangle_color(255, 255, 255), hull_color(255, 0, 0);
	string win = window_name;

	vector<Point> pt(3);
	int num_triangles = slice.triangles.size();
	for (int i = 0; i < num_triangles; i++) {
		Vec6f t = slice.triangles[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], triangle_color, 1, LINE_AA, 0);
		line(img, pt[1], pt[2], triangle_color, 1, LINE_AA, 0);
		line(img, pt[2], pt[0], triangle_color, 1, LINE_AA, 0);
	}

	imshow(win, img);
	waitKey(1);
	// Triangles in white (second)
	// Border (convex hull in blue) (last)
}

void save_frame_master(Mat &img1, Mat &img2, Size video_size, string affine, string rasterA, string rasterB) {
	MatchedGeometry geometry = read_matched_points_from_file(img1, img2, video_size);

	vector<Vec6f> trianglesA = geometry.source_geometry.triangles;
	vector<Vec6f> trianglesB = geometry.target_geometry.triangles;
	
	Rect imgA_bounds = geometry.source_geometry.img;
	Rect imgB_bounds = geometry.target_geometry.img;

	vector<vector<Point>> rastered_trianglesA = raster_triangulation(trianglesA, imgA_bounds);
	vector<vector<Point>> rastered_trianglesB = raster_triangulation(trianglesB, imgB_bounds);

	int widthA = imgA_bounds.width;
	int heightA = imgA_bounds.height;
	int widthB = imgB_bounds.width;
	int heightB = imgB_bounds.height;

	// save affine params as .csv
	// save image raster as grayscale .png from 0-65536 (2 images)
	short** gridA = grid_from_raster(widthA, heightA, rastered_trianglesA);
	short** gridB = grid_from_raster(widthB, heightB, rastered_trianglesB);
	save_raster(rasterA, gridA, widthA, heightA);
	save_raster(rasterB, gridB, widthB, heightB);

	vector<Mat> affine_forward = get_affine_transforms_forward(trianglesA, trianglesB);
	vector<Mat> affine_reverse = get_affine_transforms_reverse(trianglesB, trianglesA, affine_forward);

	float* affine_params = convert_vector_params(affine_forward, affine_reverse);
	write_float_array(affine, affine_params, trianglesA.size() * 12);
}

void trial_binary_render(uchar *image, int length, int width, int height) {
	Mat img(height, width, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			int index = (i * width + j) * 3;

			uchar r = image[index];
			uchar g = image[index + 1];
			uchar b = image[index + 2];

			Vec3b color = Vec3b(r, g, b);

			img.at<Vec3b>(i, j) = color;
		}
	}
	Mat test2 = imread("../../data_store/images/david_2.jpg");
	cout << "test";
}

void fill_top_and_bottom(Mat &img) {
	Size img_size = img.size();
	int img_width = img_size.width;
	int img_height = img_size.height;

	int midpoint = img_height / 2;

	for (int j = 0; j < img_width; j++) {
		for (int i_top = midpoint; i_top > 0; i_top--) {
			Vec3b latest = img.at<Vec3b>(i_top + 1, j);
			Vec3b current = img.at<Vec3b>(i_top, j);
			int r = current[0];
			int g = current[1];
			int b = current[2];
			if (r == 0 && g == 0 && b == 0) {
				img.at<Vec3b>(i_top, j) = latest;
			}
		}

		for (int i_bottom = midpoint; i_bottom < img_height; i_bottom++) {
			Vec3b latest = img.at<Vec3b>(i_bottom - 1, j);
			Vec3b current = img.at<Vec3b>(i_bottom, j);
			int r = current[0];
			int g = current[1];
			int b = current[2];
			if (r == 0 && g == 0 && b == 0) {
				img.at<Vec3b>(i_bottom, j) = latest;
			}
		}
	}
}

void merge_and_save(string src_path_1, string src_path_2, string dst_path) {
	Mat img_1 = imread(src_path_1, IMREAD_COLOR);
	Mat img_2 = imread(src_path_2, IMREAD_COLOR);

	// Assumptions: img_1 and img_2 are of the same size
	// The images for the video capture are the same size as img_1 and img_2
	Mat merge = merge_images(img_1, img_2);

	// checking assumptions
	Size img_size_1 = img_1.size();
	Size img_size_2 = img_2.size();
	
	if (img_size_1.width != img_size_2.width || img_size_1.height != img_size_2.height) {
		throw "Irreconcilable sizes";
	}
	Size merge_size = merge.size();
	
	Mat new_img(img_size_1.height, merge_size.width, CV_8UC3);

	int height_diff = img_size_1.height - merge_size.height;
	if (height_diff < 0) {
		throw "Not implemented";
	}

	int top_diff = 0;
	int bottom_diff = 0;
	if (height_diff % 2 != 0) {
		top_diff = height_diff / 2 + 1;
		bottom_diff = height_diff / 2;
	}
	else {
		top_diff = height_diff / 2;
		bottom_diff = height_diff / 2;
	}

	// not sure if this is working quite as I intended it
	int original_i = 0;
	for (int i = top_diff; i < merge_size.height; i++) {
		for (int j = 0; j < merge_size.width; j++) {
			new_img.at<Vec3b>(i, j) = merge.at<Vec3b>(original_i, j);
		}
		original_i++;
	}

	// fill in middle with merge (black on top and bottom)
	fill_top_and_bottom(new_img);
	save_img(dst_path, new_img);

	cout << "done";
}

void video_preprocessing(string path_1, string path_2) {
	//string input_dir = "../../data_store/flash/";
	//string input_video_1 = "judo_left.mp4";
	//string input_video_2 = "judo_right.mp4";
	int stop_frame = 5000;

	//pair<int, int> flash_result = get_flash_timing(input_dir, input_video_1, input_video_2, stop_frame);
	pair<int,int> timing_synchro = synchronize_videos(path_1, path_2, stop_frame);
	//cout << "Video 1 flash frame maxima: " << flash_result.first << endl;
	//cout << "Video 2 flash frame maxima: " << flash_result.second << endl;

	// construct two new videos from the synchronization, and save those.
	//save_trimmed_videos(flash_result, input_dir, output_dir, input_video_1, input_video_2, output_video_1, output_video_2);

}

string pad_frame_number(int frame_number) {
	// zero-padding frame number
	stringstream stream;
	stream << frame_number;
	string padded;
	stream >> padded;
	int str_length = padded.length();
	for (int i = 0; i < 6 - str_length; i++)
		padded = "0" + padded;
	return padded;
}

int video_loop(string video_path_1, string video_path_2, int start_1, int start_2){

	// do the point matching at max resolution, then rescale
	// doens't seem like we do any rescaling

	int starter_offset = 10;

	start_1 = start_1 + starter_offset;
	start_2 = start_2 + starter_offset;

	VideoCapture cap_1(video_path_1);
	VideoCapture cap_2(video_path_2);

	int num_frames_1 = cap_1.get(CV_CAP_PROP_FRAME_COUNT);
	int num_frames_2 = cap_2.get(CV_CAP_PROP_FRAME_COUNT);

	int width_1 = cap_1.get(CV_CAP_PROP_FRAME_WIDTH);
	int height_1 = cap_1.get(CV_CAP_PROP_FRAME_HEIGHT);

	int width_2 = cap_2.get(CV_CAP_PROP_FRAME_WIDTH);
	int height_2 = cap_2.get(CV_CAP_PROP_FRAME_HEIGHT);

	if (width_1 != width_2 || height_1 != height_2) {
		cout << "ERROR" << endl;
		return 0;
	}

	Size video_size = Size(width_1, height_1);

	if (!cap_1.isOpened()) {
		cout << "Video 1 failed to load." << endl;
		return -1;
	}
	if (!cap_2.isOpened()) {
		cout << "Video 2 failed to load." << endl;
	}

	cap_1.set(CV_CAP_PROP_POS_FRAMES, start_1);
	cap_2.set(CV_CAP_PROP_POS_FRAMES, start_2);

	Mat next_1;
	Mat next_2;

	int frames_remaining_1 = num_frames_1 - start_1 - 1;
	int frames_remaining_2 = num_frames_2 - start_2 - 1;
	int frames_remaining = min(frames_remaining_1, frames_remaining_2);

	// Determining how many 'jumps' are required.
	// TODO: Replace these variables with more descriptive and intuitive names.

	int renderable_frames = frames_remaining - frames_remaining % 20;
	int jump_size = 20;
	int num_jumps = renderable_frames / 20;
	int cutoff_frame = jump_size * num_jumps;

	for (int i = 0; i <= cutoff_frame; i += jump_size) {
		string padded_number = pad_frame_number(i);
		cout << "Processing frame " << i << " of " << cutoff_frame << endl;
		
		string affine_dir = "../../data_store/affine/";
		string filename_affine = "affine_" + padded_number + ".bin";

		string raster_dir = "../../data_store/raster/";
		string filename_raster_A = "raster_A_" + padded_number + ".bin";
		string filename_raster_B = "raster_B_" + padded_number + ".bin";

		string image_dir = "../../data_store/binary/";
		string filename_img_A = "imgA_" + padded_number + ".bin";
		string filename_img_B = "imgB_" + padded_number + ".bin";

		string affine = affine_dir + filename_affine;
		string rasterA = raster_dir + filename_raster_A;
		string rasterB = raster_dir + filename_raster_B;
		string imgA;
		string imgB;

		cap_1.read(next_1);
		cap_2.read(next_2);

		save_frame_master(next_1, next_2, video_size, affine, rasterA, rasterB);

		cout << "Saving image for frame " << i << endl;
		padded_number = pad_frame_number(i);
		filename_img_A = "imgA_" + padded_number + ".bin";
		filename_img_B = "imgB_" + padded_number + ".bin";
		imgA = image_dir + filename_img_A;
		imgB = image_dir + filename_img_B;
		save_img_binary(next_1, next_2, video_size, imgA, imgB);

		for (int j = 1; j < 20; j++) {
			cout << "Saving image for frame " << (i + j) << endl;
			cap_1.read(next_1);
			cap_2.read(next_2);
			padded_number = pad_frame_number(i + j);
			filename_img_A = "imgA_" + padded_number + ".bin";
			filename_img_B = "imgB_" + padded_number + ".bin";
			imgA = image_dir + filename_img_A;
			imgB = image_dir + filename_img_B;
			save_img_binary(next_1, next_2, video_size, imgA, imgB);
		}
	}
	return -1;
}

int main()
{
	// Initializing application
	cout << APPLICATION_NAME << " version " << VERSION << endl;
	cout << COMPANY_NAME << " " << COPYRIGHT_YEAR << ". " << "All rights reserved." << endl;

	// Likely, although this is subject to debate, there should be a command-line interface
	// that prompts the user to enter the video file names and other input parameters, and does error
	// checking to make sure everything is valid.

	// KEY POINT #1: The program should fail gracefully if the input data is improper!
	// KEY POINT #2: The program should exit with the first error code it encounters and print to the console. Although, for this
	// part of the program, maybe it should just go in a loop until you get it right.

	// ERROR CODE 001: File name must contain a file extension of the form .mp4, .avi, etc. (all video types we support should be clearly listed)
	// ERROR CODE 002: Video file not found. Please verify the file specified exists in the data_store/video folder (or whatever folder we use)
	string video_path_1 = "../../data_store/video/judo_left.mp4";
	string video_path_2 = "../../data_store/video/judo_right.mp4";

	// ERROR CODE 003: The 'start_offset' parameter, which specifies how many frames (if any) are to be skipped before processing the video, cannot be negative.
	// ERROR CODE 004: The 'start_offset' parameter cannot be greater than the number of frames in either of the two input videos.
	int start_offset = 500;

	// ERROR CODE 005: The 'delay' parameter must be a parsable positive floating point number (for example, 6.2657 is a valid input). 
	// ERROR CODE 006: The 'delay' parameter specified was too large and did not yield any usable frames (no overlap).
	float delay = 6.2657f;
	// ERROR CODE 007: The framerate must be a positive integer.
	int framerate = 95;

	pair<int, int> initial_offset = audio_sync(start_offset, delay, framerate);
	video_loop(video_path_1, video_path_2, initial_offset.first, initial_offset.second);

	cout << "Finished. Press enter to terminate the program.";

	cin.get();

    return 0;
}