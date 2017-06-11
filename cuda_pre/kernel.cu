
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

Mat visualize_labels(Mat labels) {
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
}

int main()
{
	//Getting the label mask from segmentation
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

	Mat only_contour = labels.clone();
	for (int i = 1; i < only_contour.rows-1; i++) {
		for (int j = 1; j < only_contour.cols-1; j++) {
			int id = only_contour.at<int>(i, j);
			if (labels.at<int>(i - 1, j - 1) != id || labels.at<int>(i - 1, j) != id || labels.at<int>(i - 1, j + 1) != id ||
				labels.at<int>(i, j - 1) != id										 || labels.at<int>(i, j + 1) != id ||
				labels.at<int>(i + 1, j - 1) != id || labels.at<int>(i + 1, j) != id || labels.at<int>(i + 1, j + 1) != id)
			{	
				only_contour.at<int>(i, j) = 0;
			}
		}
	}

	Mat label_viz = visualize_labels(labels);
	Mat contour_viz = visualize_labels(only_contour);
		
	int spx_nb = slic->getNumberOfSuperpixels();

	vector<Contour> contours_out;
	vector<Point> centers_out;
	vector<int> spx_sizes_out;
	get_spx_data(labels, spx_nb, spx_sizes_out, centers_out, contours_out);
	
	circle(result, centers_out.at(853), 2, Scalar(0, 0, 255), 2,0);


	int width = labels.size().width;
	int height = labels.size().height;

	int memsize = width * height * sizeof(int);

    return 0;
}