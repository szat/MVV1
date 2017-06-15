
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

#include <stdio.h>
#include <iostream>
#include <string>
#include <queue>

#include <binary_read.h>
#include <binary_write.h>

#include <AKAZE.h>
#include <AKAZEConfig.h>
//#include "cuda_akaze.h"
#include "cudautils.h"
#include <cuda_profiler_api.h>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;
using namespace ml;

RNG rng(12345);

typedef vector<Point> Contour;

Mat visualize_labels_int(Mat & labels) {
	Mat label_viz(labels.size(), CV_8UC3);

	int width = labels.size().width;
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			label_viz.at<Vec3b>(i, j)[0] = labels.at<int>(i, j) % 255;
			label_viz.at<Vec3b>(i, j)[1] = labels.at<int>(i, j) - labels.at<int>(i, j) % 255;
			label_viz.at<Vec3b>(i, j)[1] = labels.at<int>(i, j) / 2 % 255;
		}
	}
	return label_viz;
}

Mat visualize_labels_short(Mat & labels) {
	Mat label_viz(labels.size(), CV_8UC3);

	int width = labels.size().width;
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			label_viz.at<Vec3b>(i, j)[0] = labels.at<short>(i, j) % 255;
			label_viz.at<Vec3b>(i, j)[1] = labels.at<short>(i, j) - labels.at<short>(i, j) % 255;
			label_viz.at<Vec3b>(i, j)[1] = labels.at<short>(i, j) / 2 % 255;
		}
	}
	return label_viz;
}

Mat visualize_cluster(int cluster_id, Mat & image, vector<Contour> & contours, int spx_nb, vector<int> & spx_cluster_labels) {
	Mat viz = image.clone();
	Vec3b color;
	color[0] = 255;
	color[1] = 0;
	color[2] = 255;
	for (int i = 0; i < spx_nb; i++) {
		if (cluster_id == spx_cluster_labels.at(i)) {
			for (int j = 0; j < contours.at(i).size(); j++) {
				int row = contours.at(i).at(j).y;
				int col = contours.at(i).at(j).x;
				viz.at<Vec3b>(row, col) = color;
			}
		}
	}
	return viz;
}

int get_spx_data(Mat & labels_in, int spx_nb_in, vector<int> & sizes_out, vector<Point> & centers_out, vector<Contour> & contours_out) {
	//Stupid init

	for (int i = 0; i < spx_nb_in; i++) {
		int nb = 0;
		sizes_out.push_back(nb);

		Point center;
		center.x = 0;
		center.y = 0;
		centers_out.push_back(center);

		Contour contour;
		contours_out.push_back(contour);
	}

	//A center is an average of the positions of the pixels for a label
	for (int i = 0; i < labels_in.rows; i++) {
		for (int j = 0; j < labels_in.cols; j++) {
			int id = labels_in.at<short>(i, j);
			sizes_out.at(id)++;
			//This is correct, double checked
			centers_out.at(id).x += j;
			centers_out.at(id).y += i;
		}
	}

	for (int i = 0; i < spx_nb_in; i++) {
		centers_out.at(i).x = centers_out.at(i).x / sizes_out.at(i);
		centers_out.at(i).y = centers_out.at(i).y / sizes_out.at(i);
	}

	/*
	for (int i = 0; i < spx_nb_in; i++) {
	int col = centers_out.at(i).x;
	int row = centers_out.at(i).y;
	if (i != labels_in.at<short>(row, col)) {
	cout << "spx " << i << " has center at (" << row << " , " << col << ")" << endl;
	}
	}
	*/

	//if a center is not in the spx it represents, find a near representative
	for (int i = 0; i < spx_nb_in; i++) {
		int col = centers_out.at(i).x;
		int row = centers_out.at(i).y;
		if (i != labels_in.at<short>(row, col)) {
			Point new_center;
			int ray = (int)sqrt((float)sizes_out.at(i));
			int rows = labels_in.rows;
			int cols = labels_in.cols;

			for (int j = 0; j < ray; j++) {
				int new_row = row - j;
				int new_col = col - j;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<short>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row;
				new_col = col - j;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<short>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}
				new_row = row + 1;
				new_col = col - j;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<short>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row - 1;
				new_col = col;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<short>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}
				new_row = row + 1;
				new_col = col;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<short>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row - 1;
				new_col = col + 1;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<short>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row;
				new_col = col + 1;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<short>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row + 1;
				new_col = col + 1;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<short>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}
			}
		}
	}

	/*
	for (int i = 0; i < spx_nb_in; i++) {
	int col = centers_out.at(i).x;
	int row = centers_out.at(i).y;
	if (i != labels_in.at<short>(row, col)) {
	cout << "spx " << i << " has center at (" << row << " , " << col << ")" << endl;
	}
	}
	*/

//Finding Contours
for (int i = 1; i < labels_in.rows - 1; i++) {
	for (int j = 1; j < labels_in.cols - 1; j++) {
		int id = labels_in.at<short>(i, j);
		if (labels_in.at<short>(i - 1, j - 1) != id || labels_in.at<short>(i - 1, j) != id || labels_in.at<short>(i - 1, j + 1) != id ||
			labels_in.at<short>(i, j - 1) != id || labels_in.at<short>(i, j + 1) != id ||
			labels_in.at<short>(i + 1, j - 1) != id || labels_in.at<short>(i + 1, j) != id || labels_in.at<short>(i + 1, j + 1) != id)
		{
			Point contour_point;
			contour_point.x += j;
			contour_point.y += i;
			contours_out.at(labels_in.at<short>(i, j)).push_back(contour_point);
		}
	}
}
return 0;
}

int get_spx_means(Mat & img, Mat & labels, int spx_nb, vector<int> & sizes_in, vector<float> & c0_mean_out, vector<float> & c1_mean_out, vector<float> & c2_mean_out) {
	//Stupid init
	for (int i = 0; i < spx_nb; i++) {
		float mean = 0;
		c0_mean_out.push_back(mean);
		c1_mean_out.push_back(mean);
		c2_mean_out.push_back(mean);
	}

	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int id = labels.at<short>(i, j);
			if (id < 0 || id > spx_nb) return -1;
			else {
				c0_mean_out.at(id) += (float)img.at<Vec3b>(i, j)[0];
				c1_mean_out.at(id) += (float)img.at<Vec3b>(i, j)[1];
				c2_mean_out.at(id) += (float)img.at<Vec3b>(i, j)[2];
			}
		}
	}

	for (int i = 0; i < spx_nb; i++) {
		c0_mean_out.at(i) /= sizes_in.at(i);
		c1_mean_out.at(i) /= sizes_in.at(i);
		c2_mean_out.at(i) /= sizes_in.at(i);
	}
}

int compute_and_save_spx(string img_path, string save_path) {
	Mat img = imread(img_path);
	if (img.empty())
	{
		cout << "Could not open image..." << img_path << "\n";
		return -1;
	}

	Mat converted;
	cvtColor(img, converted, COLOR_BGR2HSV);

	int algorithm = 0;
	int region_size = 15;
	int ruler = 45;
	int min_element_size = 20;
	int num_iterations = 8;

	cout << "New computation!" << endl;

	Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(converted, algorithm + SLIC, region_size, float(ruler));
	slic->iterate(num_iterations);
	if (min_element_size > 0) 	slic->enforceLabelConnectivity(min_element_size);

	//Mat mask;
	//slic->getLabelContourMask(mask, true);

	Mat labels;
	slic->getLabels(labels);

	int spx_nb = slic->getNumberOfSuperpixels();

	//The biggest ID of superpixel will be spx_nb-1, swap that one with the superpixel at position (0,0)

	labels.setTo(spx_nb + 1, labels == spx_nb - 1);
	int temp = labels.at<int>(0, 0);
	labels.setTo(spx_nb - 1, labels == labels.at<int>(0, 0));
	labels.setTo(temp, labels == spx_nb + 1);

	Mat labels_short;

	labels.convertTo(labels_short, CV_16U);
	Mat labels_flat = labels_short.reshape(1, 1);
	int nb_px = labels_short.rows  * labels_short.cols;

	short* spx_data = reinterpret_cast<short*>(labels_flat.data);
	write_short_array(save_path, spx_data, nb_px);

	return 0;
}

int main()
{
	//TOGGLE THIS, no catch code yet
	//compute_and_save_spx(img_path, save_path);

	//Getting the label mask from segmentation
	string img_path_1 = "..\\data_store\\images\\c1_img_000177.png";
	string save_path_1 = "..\\data_store\\spx\\c1_img_000177_spx.bin";
	Mat img_1;
	int width_1 = img_1.size().width;
	int height_1 = img_1.size().height;
	int nb_px_1 = width_1 & height_1;
	img_1 = imread(img_path_1);
	if (img_1.empty())
	{
		cout << "Could not open image..." << img_path_1 << "\n";
		return -1;
	}
	short * spx_data_1 = read_short_array(save_path_1, nb_px_1);
	int spx_nb_1 = spx_data_1[0] + 1;
	Mat labels_1 = Mat(size(img_1), CV_16U, spx_data_1);

	string img_path_2 = "..\\data_store\\images\\c2_img_000177.png";
	string save_path_2 = "..\\data_store\\spx\\c2_img_000177_spx.bin";
	Mat img_2;
	int width_2 = img_2.size().width;
	int height_2 = img_2.size().height;
	int nb_px_2 = width_2 & height_2;
	img_2 = imread(img_path_2);
	if (img_2.empty())
	{
		cout << "Could not open image..." << img_path_2 << "\n";
		return -1;
	}
	short * spx_data_2 = read_short_array(save_path_2, nb_px_2);
	int spx_nb_2 = spx_data_2[0] + 1;
	Mat labels_2 = Mat(size(img_2), CV_16U, spx_data_2);

	cout << "biggest index spx_nb_1 " << spx_nb_1 << endl;
	cout << "biggest index spx_nb_2 " << spx_nb_2 << endl;

	vector<Contour> contours_out_1;
	vector<Point> centers_out_1;
	vector<int> spx_sizes_out_1;
	get_spx_data(labels_1, spx_nb_1, spx_sizes_out_1, centers_out_1, contours_out_1);
	Mat img_hsl_1;
	cvtColor(img_1, img_hsl_1, COLOR_BGR2HLS);
	vector<float> c0_mean_out_1;
	vector<float> c1_mean_out_1;
	vector<float> c2_mean_out_1;
	get_spx_means(img_1, labels_1, spx_nb_1, spx_sizes_out_1, c0_mean_out_1, c1_mean_out_1, c2_mean_out_1);
	Mat c0_mat_1 = Mat(c0_mean_out_1.size(), 1, CV_32FC1);
	memcpy(c0_mat_1.data, c0_mean_out_1.data(), c0_mean_out_1.size() * sizeof(float));
	Mat c1_mat_1 = Mat(c1_mean_out_1.size(), 1, CV_32FC1);
	memcpy(c1_mat_1.data, c1_mean_out_1.data(), c1_mean_out_1.size() * sizeof(float));
	Mat c2_mat_1 = Mat(c2_mean_out_1.size(), 1, CV_32FC1);
	memcpy(c2_mat_1.data, c2_mean_out_1.data(), c2_mean_out_1.size() * sizeof(float));
	Mat samples_1;
	hconcat(c0_mat_1, c2_mat_1, samples_1);

	cout << "nb rows samples_1 " << samples_1.rows << endl;
	cout << "nb cols samples_1 " << samples_1.cols << endl;

	vector<Contour> contours_out_2;
	vector<Point> centers_out_2;
	vector<int> spx_sizes_out_2;
	get_spx_data(labels_2, spx_nb_2, spx_sizes_out_2, centers_out_2, contours_out_2);
	Mat img_hsl_2;
	cvtColor(img_2, img_hsl_2, COLOR_BGR2HLS);
	vector<float> c0_mean_out_2;
	vector<float> c1_mean_out_2;
	vector<float> c2_mean_out_2;
	get_spx_means(img_2, labels_2, spx_nb_2, spx_sizes_out_2, c0_mean_out_2, c1_mean_out_2, c2_mean_out_2);
	Mat c0_mat_2 = Mat(c0_mean_out_2.size(), 1, CV_32FC1);
	memcpy(c0_mat_2.data, c0_mean_out_2.data(), c0_mean_out_2.size() * sizeof(float));
	Mat c1_mat_2 = Mat(c1_mean_out_2.size(), 1, CV_32FC1);
	memcpy(c1_mat_2.data, c1_mean_out_2.data(), c1_mean_out_2.size() * sizeof(float));
	Mat c2_mat_2 = Mat(c2_mean_out_2.size(), 1, CV_32FC1);
	memcpy(c2_mat_2.data, c2_mean_out_2.data(), c2_mean_out_2.size() * sizeof(float));
	Mat samples_2;
	hconcat(c0_mat_2, c2_mat_2, samples_2);

	cout << "nb rows samples_2 " << samples_2.rows << endl;
	cout << "nb cols samples_2 " << samples_2.cols << endl;

	//Makes samples_1 and samples_2 into one
	Mat samples_total;
	vconcat(samples_1, samples_2, samples_total);

	cout << "nb rows samples_total " << samples_total.rows << endl;
	cout << "nb cols samples_total " << samples_total.cols << endl;

	//k-means
	Mat cluster_labels_total;
	Mat centers_total;
	kmeans(samples_total, 10, cluster_labels_total, TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 50, 0.0001), 10, KMEANS_PP_CENTERS, centers_total);
	
	
	vector<int> spx_labels_1;
	for (int i = 0; i < spx_nb_1; i++) {
		spx_labels_1.push_back(cluster_labels_total.at<int>(i, 0));
	}
	vector<int> spx_labels_2;
	for (int i = spx_nb_1; i < spx_nb_1 + spx_nb_2; i++) {
		spx_labels_2.push_back(cluster_labels_total.at<int>(i, 0));
	}

	//Would be useful to send back cluster_labels_total into cluster_labels of sizes corresponding to the initial image
	cout << "spx_nb_1 " << spx_nb_1 << endl;
	cout << "spx_nb_2 " << spx_nb_2 << endl;

	//Mat spx_labels_1_mat = cluster_labels_total(Rect(0, 0, 1, spx_nb_1));
	//Mat spx_labels_2_mat = cluster_labels_total(Rect(0, spx_nb_1, 1, spx_nb_2));

	/*
	namedWindow("C1", WINDOW_NORMAL);
	namedWindow("C2", WINDOW_NORMAL);

	cv::resizeWindow("C1", 1536/2, 864);
	cv::resizeWindow("C2", 1536/2, 864);

	Mat viz_1 = img_1.clone();
	Mat viz_2 = img_2.clone();
	for (int i = 0; i < centers_total.rows; i++) {
		viz_1 = visualize_cluster(i, img_1, contours_out_1, spx_nb_1, spx_labels_1);
		viz_2 = visualize_cluster(i, img_2, contours_out_2, spx_nb_2, spx_labels_2);
		imshow("C1", viz_1);
		imshow("C2", viz_2);
		cout << "We are displaying cluster tag " << i << "." << endl;
		waitKey(0);
	}
	*/

	//For a certain id or group of id construct the mask for img_1 and img_2

	int id = 0;

	Mat roi_1 = Mat(img_1.size(), CV_8UC1);
	roi_1.setTo(0);
	for (int i = 0; i < labels_1.rows; i++) {
		for (int j = 0; j < labels_1.cols; j++) {
			int spx_id = labels_1.at<short>(i, j);
			int cluster_id = spx_labels_1.at(spx_id);
			if (cluster_id == id) {
				roi_1.at<uchar>(i, j) = 1;
			}
		}
	}

	Mat roi_2 = Mat(img_2.size(), CV_8UC1);
	roi_2.setTo(0);
	for (int i = 0; i < labels_2.rows; i++) {
		for (int j = 0; j < labels_2.cols; j++) {
			int spx_id = labels_2.at<short>(i, j);
			int cluster_id = spx_labels_2.at(spx_id);
			if (cluster_id == id) {
				roi_2.at<uchar>(i, j) = 1;
			}
		}
	}

	/*
	//AKAZE CODE
	AKAZEOptions options;
	cv::Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
	string img_path1 = "C:/Users/Adrian/Documents/GitHub/mvv/data_store/images/david_1.jpg";
	string img_path2 = "C:/Users/Adrian/Documents/GitHub/mvv/data_store/images/david_2.jpg";
	float ratio = 0.0, rfactor = .60;
	int nkpts1 = 0, nkpts2 = 0, nmatches = 0, ninliers = 0, noutliers = 0;

	vector<cv::KeyPoint> kpts1, kpts2;
	vector<vector<cv::DMatch> > dmatches;
	cv::Mat desc1, desc2;
	cv::Mat HG;

	// Variables for measuring computation times
	double t1 = 0.0, t2 = 0.0;
	double takaze = 0.0, tmatch = 0.0;

	// Parse the input command line options
	//if (parse_input_options(options,img_path1,img_path2,homography_path,argc,argv))
	//  return -1;

	// Read image 1 and if necessary convert to grayscale.
	img1 = cv::imread(img_path1, 0);
	if (img1.data == NULL) {
	cerr << "Error loading image 1: " << img_path1 << endl;
	return -1;
	}

	// Read image 2 and if necessary convert to grayscale.
	img2 = cv::imread(img_path2, 0);
	if (img2.data == NULL) {
	cerr << "Error loading image 2: " << img_path2 << endl;
	return -1;
	}

	// Read ground truth homography file
	bool use_ransac = true;
	//if (read_homography(homography_path, HG) == false)
	//  use_ransac = true;

	// Convert the images to float
	img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
	img2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

	// Color images for results visualization
	img1_rgb = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
	img2_rgb = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
	img_com = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);
	img_r = cv::Mat(cv::Size(img_com.cols*rfactor, img_com.rows*rfactor), CV_8UC3);

	// Create the first AKAZE object
	options.img_width = img1.cols;
	options.img_height = img1.rows;
	libAKAZECU::AKAZE evolution1(options);

	// Create the second HKAZE object
	options.img_width = img2.cols;
	options.img_height = img2.rows;
	libAKAZECU::AKAZE evolution2(options);

	t1 = cv::getTickCount();

	cudaProfilerStart();

	evolution1.Create_Nonlinear_Scale_Space(img1_32);
	evolution1.Feature_Detection(kpts1);
	evolution1.Compute_Descriptors(kpts1, desc1);

	evolution2.Create_Nonlinear_Scale_Space(img2_32);
	evolution2.Feature_Detection(kpts2);
	evolution2.Compute_Descriptors(kpts2, desc2);

	t2 = cv::getTickCount();
	takaze = 1000.0*(t2 - t1) / cv::getTickFrequency();

	// Show matching statistics
	cout << "Number of Keypoints Image 1: " << nkpts1 << endl;
	cout << "Number of Keypoints Image 2: " << nkpts2 << endl;
	cout << "A-KAZE Features Extraction Time (ms): " << takaze << endl;
	*/


	//best_labels is the classification

	//k-means
	/*
	Mat best_labels;
	Mat centers;
	kmeans(samples_1, 10, best_labels, TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 50, 0.0001), 10, KMEANS_PP_CENTERS, centers);
	//best_labels is the classification 

	vector<int> spx_labels;
	spx_labels.assign((int*)best_labels.datastart, (int*)best_labels.dataend);


	vector<Contour> contours_out;
	vector<Point> centers_out;
	vector<int> spx_sizes_out;
	get_spx_data(labels_2, spx_nb_2, spx_sizes_out_2, centers_out_2, contours_out_2);


	//Clustering code, we will attempt to group superpixels that belong to each other color wise
	//First convert the color space to Lab for instance, so that the luminosity can be ignored. 
	//Second blur the image
	//Do clustering, for instance K-means (could evventually use the gap statistic) or GMM/EM

	//img is loaded, lets work with that

	Mat img_hsl;
	cvtColor(img_1, img_hsl, COLOR_BGR2HLS);
	
	vector<float> c0_mean_out;
	vector<float> c1_mean_out;
	vector<float> c2_mean_out;

	get_spx_means(img_1, labels_1, spx_nb_1, spx_sizes_out_1, c0_mean_out, c1_mean_out, c2_mean_out);

	Mat c0_mat = Mat(c0_mean_out.size(), 1, CV_32FC1);
	memcpy(c0_mat.data, c0_mean_out.data(), c0_mean_out.size() * sizeof(float));
	Mat c1_mat = Mat(c1_mean_out.size(), 1, CV_32FC1);
	memcpy(c1_mat.data, c1_mean_out.data(), c1_mean_out.size() * sizeof(float));
	Mat c2_mat = Mat(c2_mean_out.size(), 1, CV_32FC1);
	memcpy(c2_mat.data, c2_mean_out.data(), c2_mean_out.size() * sizeof(float));

	Mat samples;
	hconcat(c0_mat, c2_mat, samples);
	
	cout << "nb rows " << samples.rows << endl;
	cout << "nb cols " << samples.cols << endl;

	//k-means
	Mat best_labels;
	Mat centers;
	kmeans(samples, 10, best_labels, TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 50, 0.0001), 10, KMEANS_PP_CENTERS, centers);
	//best_labels is the classification 

	vector<int> spx_labels;
	spx_labels.assign((int*)best_labels.datastart, (int*)best_labels.dataend);

	Mat img_viz2 = img_1.clone();
	for (int c = 0; c < 10; c++) {
		for (int p = 0; p < spx_labels.size(); p++) {
			int id = spx_labels.at(p);
			if (c == id) {
				for (int ii = 0; ii < contours_out.at(p).size(); ii++) {
					int x = contours_out.at(p).at(ii).x;
					int y = contours_out.at(p).at(ii).y;
					Vec3b color; color[0] = 20*c; color[1] = 10*c*c %255; color[2] = 5*c;
					img_viz2.at<Vec3b>(y, x) = color;
				}
			}
		}
	}

	Mat vix = visualize_cluster(7, img_1, contours_out, spx_nb_1, spx_labels);
	
	//best_labels.at<int>(1, 10);
	*/

	//GMM
	/*
	//20000 points still acceptable time
	Ptr<ml::EM> em = ml::EM::create();
	bool status = em->isTrained();
	status = em->trainEM(samples);
	cout << "cluster number " << em->getClustersNumber() << endl;
	status = em->isTrained();
	*/

	/*
	//AKAZE CODE
	AKAZEOptions options;
	cv::Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
	string img_path1 = "C:/Users/Adrian/Documents/GitHub/mvv/data_store/images/david_1.jpg";
	string img_path2 = "C:/Users/Adrian/Documents/GitHub/mvv/data_store/images/david_2.jpg";
	float ratio = 0.0, rfactor = .60;
	int nkpts1 = 0, nkpts2 = 0, nmatches = 0, ninliers = 0, noutliers = 0;

	vector<cv::KeyPoint> kpts1, kpts2;
	vector<vector<cv::DMatch> > dmatches;
	cv::Mat desc1, desc2;
	cv::Mat HG;

	// Variables for measuring computation times
	double t1 = 0.0, t2 = 0.0;
	double takaze = 0.0, tmatch = 0.0;

	// Parse the input command line options
	//if (parse_input_options(options,img_path1,img_path2,homography_path,argc,argv))
	//  return -1;

	// Read image 1 and if necessary convert to grayscale.
	img1 = cv::imread(img_path1, 0);
	if (img1.data == NULL) {
		cerr << "Error loading image 1: " << img_path1 << endl;
		return -1;
	}

	// Read image 2 and if necessary convert to grayscale.
	img2 = cv::imread(img_path2, 0);
	if (img2.data == NULL) {
		cerr << "Error loading image 2: " << img_path2 << endl;
		return -1;
	}

	// Read ground truth homography file
	bool use_ransac = true;
	//if (read_homography(homography_path, HG) == false)
	//  use_ransac = true;

	// Convert the images to float
	img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);
	img2.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

	// Color images for results visualization
	img1_rgb = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
	img2_rgb = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
	img_com = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);
	img_r = cv::Mat(cv::Size(img_com.cols*rfactor, img_com.rows*rfactor), CV_8UC3);

	// Create the first AKAZE object
	options.img_width = img1.cols;
	options.img_height = img1.rows;
	libAKAZECU::AKAZE evolution1(options);

	// Create the second HKAZE object
	options.img_width = img2.cols;
	options.img_height = img2.rows;
	libAKAZECU::AKAZE evolution2(options);

	t1 = cv::getTickCount();

	cudaProfilerStart();

	evolution1.Create_Nonlinear_Scale_Space(img1_32);
	evolution1.Feature_Detection(kpts1);
	evolution1.Compute_Descriptors(kpts1, desc1);

	evolution2.Create_Nonlinear_Scale_Space(img2_32);
	evolution2.Feature_Detection(kpts2);
	evolution2.Compute_Descriptors(kpts2, desc2);

	t2 = cv::getTickCount();
	takaze = 1000.0*(t2 - t1) / cv::getTickFrequency();

	// Show matching statistics
	cout << "Number of Keypoints Image 1: " << nkpts1 << endl;
	cout << "Number of Keypoints Image 2: " << nkpts2 << endl;
	cout << "A-KAZE Features Extraction Time (ms): " << takaze << endl;
	*/
	return 0;
}
