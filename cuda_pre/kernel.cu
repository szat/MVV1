
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <string>
#include <queue>

#include <binary_read.h>
#include <binary_write.h>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

__global__
void slic_segmentation(uchar3* cielab, int* labels, int width, int height, int spx_nb, int compactness, int metric) {
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int i = (row * width + col);

	// should not need to do this check if everything is good, must be an extra pixel
	if (i >= width * height) return;
	if ((row >= height) || (col >= width)) return;

	uchar3 blank = uchar3();
	blank.x = 0;
	blank.y = 0;
	blank.z = 0;
	cielab[i] = blank;
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

Mat visualize_labels_int(Mat labels) {
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

Mat visualize_labels_short(Mat labels) {
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

typedef vector<Point> Contour;

void get_spx_data(Mat & labels_in, int spx_nb_in, vector<int> & sizes_out, vector<Point> & centers_out, vector<Contour> & contours_out) {
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
			int id = labels_in.at<int>(i, j);
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
		if (i != labels_in.at<int>(row, col)) {
			cout << "spx " << i << " has center at (" << row << " , " << col << ")" << endl;
		}
	}
	*/

	//if a center is not in the spx it represents, find a near representative
	for (int i = 0; i < spx_nb_in; i++) {
		int col = centers_out.at(i).x;
		int row = centers_out.at(i).y;
		if (i != labels_in.at<int>(row, col)) {
			Point new_center;
			int ray = (int)sqrt((float)sizes_out.at(i));
			int rows = labels_in.rows;
			int cols = labels_in.cols;

			for (int j = 0; j < ray; j++) {
				int new_row = row - j;
				int new_col = col - j;			
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<int>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row;
				new_col = col - j;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<int>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}
				new_row = row + 1;
				new_col = col - j;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<int>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row - 1;
				new_col = col;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<int>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}
				new_row = row + 1;
				new_col = col;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<int>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row - 1;
				new_col = col + 1;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<int>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row;
				new_col = col + 1;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<int>(new_row, new_col)) {
						new_center.x = new_col;
						new_center.y = new_row;
						centers_out.at(i) = new_center;
						break;
					}
				}

				new_row = row + 1;
				new_col = col + 1;
				if (new_row >= 0 && new_row < rows && new_col >= 0 && new_col < cols) {
					if (i == labels_in.at<int>(new_row, new_col)) {
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
		if (i != labels_in.at<int>(row, col)) {
			cout << "spx " << i << " has center at (" << row << " , " << col << ")" << endl;
		}
	}
	*/

	//Finding Contours
	for (int i = 1; i < labels_in.rows - 1; i++) {
		for (int j = 1; j < labels_in.cols - 1; j++) {
			int id = labels_in.at<int>(i, j);
			if (labels_in.at<int>(i - 1, j - 1) != id || labels_in.at<int>(i - 1, j) != id || labels_in.at<int>(i - 1, j + 1) != id ||
				labels_in.at<int>(i, j - 1) != id || labels_in.at<int>(i, j + 1) != id ||
				labels_in.at<int>(i + 1, j - 1) != id || labels_in.at<int>(i + 1, j) != id || labels_in.at<int>(i + 1, j + 1) != id)
			{
				Point contour_point;
				contour_point.x += j;
				contour_point.y += i;
				contours_out.at(labels_in.at<int>(i, j)).push_back(contour_point);
			}
		}
	}
}

int compute_and_save_spx(string img_file,  string spx_name) {
	Mat img = imread(img_file);
	if (img.empty())
	{
		cout << "Could not open image..." << img_file << "\n";
		return -1;
	}

	string save_path = "C:/Users/Adrian/Documents/GitHub/mvv/data_store/spx/" + spx_name;

	Mat converted;
	cvtColor(img, converted, COLOR_BGR2HSV);

	int algorithm = 0;
	int region_size = 25;
	int ruler = 45;
	int min_element_size = 50;
	int num_iterations = 6;

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
	compute_and_save_spx("..\\data_store\\images\\david_1.jpg", "david_1.bin");

	string img_file = "..\\data_store\\images\\david_1.jpg";
	Mat img;

	img = imread(img_file);
	if (img.empty())
	{
		cout << "Could not open image..." << img_file << "\n";
		return -1;
	}

	int nb_px = img.rows * img.cols;
	
	short * spx_data_2 = read_short_array("C:/Users/Adrian/Documents/GitHub/mvv/data_store/spx/david_1.bin", nb_px);

	Mat loaded = Mat(size(img), CV_16U, spx_data_2);

	//Mat label_viz = visualize_labels_int(labels);
	Mat loaded_viz = visualize_labels_short(loaded);

	int spx_nb = (int)loaded.at<short>(0, 0)+1;

	cout << "biggest index " << spx_nb << endl;

	vector<Contour> contours_out;
	vector<Point> centers_out;
	vector<int> spx_sizes_out;

	get_spx_data(loaded, spx_nb, spx_sizes_out, centers_out, contours_out);

	/*
	string img_file = "..\\data_store\\images\\david_1.jpg";
	Mat img;

	img = imread(img_file);
	if (img.empty())
	{
		cout << "Could not open image..." << img_file << "\n";
		return -1;
	}
	
	Mat converted;
	cvtColor(img, converted, COLOR_BGR2HSV);

	int algorithm = 0;
	int region_size = 25;
	int ruler = 45;
	int min_element_size = 50;
	int num_iterations = 6;

	cout << "New computation!" << endl;

	Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(converted, algorithm + SLIC, region_size, float(ruler));
	slic->iterate(num_iterations);
	if (min_element_size > 0) 	slic->enforceLabelConnectivity(min_element_size);

	Mat result, mask;
	result = img.clone();
	slic->getLabelContourMask(mask, true);

	result.setTo(Scalar(0, 255, 0), mask);

	Mat labels;
	slic->getLabels(labels);

	Mat labels_short;

	labels.convertTo(labels_short, CV_16U);
	Mat labels_flat = labels_short.reshape(1, 1);
	int nb_px = labels_short.rows  * labels_short.cols;

	short* spx_data = reinterpret_cast<short*>(labels_flat.data);
	write_short_array("C:/Users/Adrian/Documents/GitHub/mvv/data_store/spx/spx_1.bin", spx_data, nb_px);

	short * spx_data_2 = read_short_array("C:/Users/Adrian/Documents/GitHub/mvv/data_store/spx/spx_1.bin", nb_px);

	Mat loaded = Mat(size(labels), CV_16U, spx_data_2); 

	Mat label_viz = visualize_labels_int(labels);
	Mat loaded_viz = visualize_labels_short(loaded);
		
	int spx_nb = slic->getNumberOfSuperpixels();

	cout << "number of superpixels" << spx_nb;
	double min, max;
	cv::minMaxLoc(loaded, &min, &max);
	cout << "max " << max << endl;
	cout << "min" << min << endl;

	vector<Contour> contours_out;
	vector<Point> centers_out;
	vector<int> spx_sizes_out;
	get_spx_data(labels, spx_nb, spx_sizes_out, centers_out, contours_out);

	*/

	/*
	circle(result, centers_out.at(309), 2, Scalar(0, 255, 255), 2, 0);
	for (int i = 0; i < contours_out.at(309).size(); i++) {
		circle(result, contours_out.at(309).at(i), 1, Scalar(0, 0, 255), 1, 0);
	}
	*/

	/*
	int width = labels.size().width;
	int height = labels.size().height;

	int memsize = width * height * sizeof(int);

	*/
    return 0;
}