
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
	//Getting the label mask from segmentation
	string img_path = "..\\data_store\\images\\c1_img_000177.png";
	string save_path = "..\\data_store\\spx\\c1_img_000177_spx.bin";

	//TOGGLE THIS, no catch code yet
	//compute_and_save_spx(img_path, save_path);

	Mat img;
	int width = img.size().width;
	int height = img.size().height;
	int nb_px = width & height;

	img = imread(img_path);
	if (img.empty())
	{
		cout << "Could not open image..." << img_path << "\n";
		return -1;
	}

	short * spx_data = read_short_array(save_path, nb_px);
	int spx_nb = spx_data[0] + 1;

	Mat labels = Mat(size(img), CV_16U, spx_data);
	Mat labels_viz = visualize_labels_short(labels);

	cout << "biggest index " << spx_nb << endl;

	vector<Contour> contours_out;
	vector<Point> centers_out;
	vector<int> spx_sizes_out;
	get_spx_data(labels, spx_nb, spx_sizes_out, centers_out, contours_out);

	//To draw all the spx boundaries
	Mat img_viz = img.clone();
	for (int i = 0; i < contours_out.size(); i++) {
		for (int ii = 0; ii < contours_out.at(i).size(); ii++) {
			int x = contours_out.at(i).at(ii).x;
			int y = contours_out.at(i).at(ii).y;
			Vec3b color; color[0] = 0; color[1] = 0; color[2] = 0;
			img_viz.at<Vec3b>(y, x) = color;
		}
	}

	//To visualize a single spx
	int spx_id = 800;
	circle(img_viz, centers_out.at(spx_id), 2, Scalar(0, 0, 255), 1, 0);
	for (int ii = 0; ii < contours_out.at(spx_id).size(); ii++) {
		int x = contours_out.at(spx_id).at(ii).x;
		int y = contours_out.at(spx_id).at(ii).y;
		Vec3b color;
		color[0] = 0;
		color[1] = 0;
		color[2] = 255;
		img_viz.at<Vec3b>(y, x) = color;
	}

	//Clustering code, we will attempt to group superpixels that belong to each other color wise
	//First convert the color space to Lab for instance, so that the luminosity can be ignored. 
	//Second blur the image
	//Do clustering, for instance K-means (could evventually use the gap statistic) or GMM/EM

	//img is loaded, lets work with that

	Mat img_hsl;
	cvtColor(img, img_hsl, COLOR_BGR2HLS);
	
	vector<float> c0_mean_out;
	vector<float> c1_mean_out;
	vector<float> c2_mean_out;

	get_spx_means(img, labels, spx_nb, spx_sizes_out, c0_mean_out, c1_mean_out, c2_mean_out);

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

	Mat img_viz2 = img.clone();
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

	Mat vix = visualize_cluster(7, img, contours_out, spx_nb, spx_labels);
	
	//best_labels.at<int>(1, 10);

	//GMM
	/*
	//20000 points still acceptable time
	Ptr<ml::EM> em = ml::EM::create();
	bool status = em->isTrained();
	status = em->trainEM(samples);
	cout << "cluster number " << em->getClustersNumber() << endl;
	status = em->isTrained();
	*/

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

	return 0;
}
