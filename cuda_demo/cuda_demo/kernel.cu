#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <conio.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "binary_io.h"

using namespace std;
using namespace cv;

typedef std::chrono::high_resolution_clock Clock;

__global__
void kernel2D(uchar* d_output, uchar* d_input, int w, int h, float * d_affineData)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c;

	if ((r >= h) || (c >= w)) return;

	int new_c = (int)(d_affineData[0] * (float)c + d_affineData[1] * (float)r + d_affineData[2]);
	int new_r = (int)(d_affineData[3] * (float)c + d_affineData[4] * (float)r + d_affineData[5]);

	if ((new_r >= h) || (new_c >= w) || (new_r < 0) || (new_c < 0)) return;

	int new_i = new_r * w + new_c;
	d_output[new_i] = d_input[i];
}

__global__
void kernel2D_subpix(uchar* d_output, uchar* d_input, short* d_raster1, int w, int h, float * d_affineData, int subDiv, float tau, bool reverse)
{
	if (tau >= 1 || tau < 0) return;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int i = (row * w + col) * 3;

	// should not need to do this check if everything is good, must be an extra pixel
	if (i + 2 >= w * h * 3) return;

	d_output[i] = 0;
	d_output[i + 1] = 0;
	d_output[i+2] = d_input[i+2];

	//if ((r >= h) || (c >= w)) return;

	/*
	short affine_index = d_raster1[i];
	short offset = affine_index * 12;
	if (reverse) {
		offset += 6;
	}
	if (affine_index != 0) {
		float diff = 1 / (float)subDiv;
		for (int i = 0; i < subDiv; i++) {
			for (int j = 0; j < subDiv; j++) {
				int new_c = (int)(((1-tau) + tau*d_affineData[offset]) * (float)(c - 0.5 + (diff * i)) + (tau * d_affineData[offset + 1]) * (float)(r - 0.5 + (diff * j)) + (tau * d_affineData[offset + 2]));
				int new_r = (int)((tau * d_affineData[offset + 3]) * (float)(c - 0.5 + (diff * i)) + ((1-tau) + tau * d_affineData[offset + 4]) * (float)(r - 0.5 + (diff * j)) + (tau * d_affineData[offset + 5]));
				if ((new_r >= h) || (new_c >= w) || (new_r < 0) || (new_c < 0)) return;
				int new_i = new_r * w + new_c;
				int new_index = new_i * 3;
				d_output[new_index] = channel_1;
				d_output[new_index + 1] = channel_2;
				d_output[new_index + 2] = channel_3;
			}
		}
	}
	*/
}

__global__
void kernel2D_add(uchar* d_output, uchar* d_input_1, uchar* d_input_2, int w, int h, float tau) {
	//tau is from a to b
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int index = (r * w + c);

	if ((r >= h) || (c >= w)) return;

	/*
	for (int i = 0; i < 3; i++) {
		if (d_input_1[index + i] == 0) {
			d_output[index + i] = d_input_2[index + i];
		}
		else if (d_input_2[index + i] == 0) {
			d_output[index + i] = d_input_1[index + i];
		}
		else {
			d_output[index + i] = tau*d_input_1[index + i] + (1 - tau)*d_input_2[index + i];
		}
	}
	*/
	d_output[index] = (uchar)200;
	d_output[index + 1] = (uchar)200;
	d_output[index + 2] = (uchar)200;

}

void trial_binary_render(uchar *image, int width, int height) {
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

int main(int argc, char ** argv) {
	cout << "welcome to cuda_demo testing unit!" << endl;
	cout << "loading 2 images with openCV, processing and adding them with cuda (grayscale)." << endl;

	// Initializing CUDA
	uchar *h_tester = new uchar[1];
	h_tester[0] = (uchar)0;
	uchar *d_tester;
	cudaMalloc((void**)&d_tester, sizeof(uchar));
	cudaMemcpy(d_tester, h_tester, sizeof(uchar), cudaMemcpyHostToDevice);
	cudaFree(d_tester);

	auto t1 = std::chrono::high_resolution_clock::now();

	string img_path_1 = "../../data_store/binary/david_1.bin";
	string img_path_2 = "../../data_store/binary/david_2.bin";
	string raster1_path = "../../data_store/raster/rasterA.bin";
	string raster2_path = "../../data_store/raster/rasterB.bin";
	string affine_path = "../../data_store/affine/affine_1.bin";

	// BINARY IMAGE READ
	int length_1 = 0;
	int length_2 = 0;
	int width_1 = 0;
	int width_2 = 0;
	int height_1 = 0;
	int height_2 = 0;
	uchar *h_in_1 = read_uchar_array(img_path_1, length_1, width_1, height_1);
	uchar *h_in_2 = read_uchar_array(img_path_2, length_2, width_2, height_2);
	
	// RASTER READ
	int num_pixels_1 = 0;
	int num_pixels_2 = 0;
	short *h_raster1 = read_short_array(raster1_path, num_pixels_1);
	short *h_raster2 = read_short_array(raster2_path, num_pixels_2);

	// AFFINE READ
	int num_floats = 0;
	float *h_affine_data = read_float_array(affine_path, num_floats);
	int num_triangles = num_floats / 12;

	if (height_1 != height_2 || width_1 != width_2) {
		cout << "Incompatible image sizes. Program will now crash.\n";
		exit(-1);
	}

	int W = width_1;
	int H = height_1;
	int mem_alloc = W * H * 3 * sizeof(uchar);

	uchar *h_out_1;
	uchar *h_out_2;
	uchar *h_sum;
	
	// there must be a faster way to initialize these arrays to all zeros

	h_out_1 = (uchar*)malloc(mem_alloc);
	for (int j = 0; j < W*H*3; j++) h_out_1[j] = 0;

	h_out_2 = (uchar*)malloc(mem_alloc);
	for (int j = 0; j < W*H*3; j++) h_out_2[j] = 0;

	h_sum = (uchar*)malloc(mem_alloc);
	for (int j = 0; j < W*H*3; j++) h_sum[j] = 0;
	
	//--Sending the data to the GPU memory
	cout << "declaring device data-structures..." << endl;

	float * d_affine_data;
	cudaMalloc((void**)&d_affine_data, num_floats * sizeof(float));
	cudaMemcpy(d_affine_data, h_affine_data, num_floats * sizeof(float), cudaMemcpyHostToDevice);

	short *d_raster1;
	cudaMalloc((void**)&d_raster1, W * H * sizeof(short));
	cudaMemcpy(d_raster1, h_raster1, W * H * sizeof(short), cudaMemcpyHostToDevice);

	short *d_raster2;
	cudaMalloc((void**)&d_raster2, W * H * sizeof(short));
	cudaMemcpy(d_raster2, h_raster2, W * H * sizeof(short), cudaMemcpyHostToDevice);

	uchar * d_in_1;
	cudaMalloc((void**)&d_in_1, mem_alloc);
	cudaMemcpy(d_in_1, h_in_1, mem_alloc, cudaMemcpyHostToDevice);

	uchar * d_out_1;
	cudaMalloc((void**)&d_out_1, mem_alloc);
	cudaMemcpy(d_out_1, h_out_1, mem_alloc, cudaMemcpyHostToDevice);

	uchar * d_in_2;
	cudaMalloc((void**)&d_in_2, mem_alloc);
	cudaMemcpy(d_in_2, h_in_2, mem_alloc, cudaMemcpyHostToDevice);

	uchar * d_out_2;
	cudaMalloc((void**)&d_out_2, mem_alloc);
	cudaMemcpy(d_out_2, h_out_2, mem_alloc, cudaMemcpyHostToDevice);

	uchar * d_sum;
	cudaMalloc((void**)&d_sum, mem_alloc);
	cudaMemcpy(d_sum, h_sum, mem_alloc, cudaMemcpyHostToDevice);

	dim3 blockSize(32, 32);
	int bx = (W + 32 - 1) / 32;
	int by = (H + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);

	float tau = 0.5f;
	float reverse_tau = 1.0f - tau;
	int reversal_offset = 0;


	kernel2D_subpix << <gridSize, blockSize >> >(d_out_1, d_in_1, d_raster1, W, H, d_affine_data, 4, tau, false);
	//kernel2D_subpix << <gridSize, blockSize >> >(d_out_2, d_in_2, d_raster2, W, H, d_affine_data, 4, reverse_tau, true);
	//kernel2D_add << <gridSize, blockSize >> > (d_sum, d_out_1, d_out_1, W, H, tau);

	cudaMemcpy(h_out_1, d_out_1, mem_alloc, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_out_2, d_out_2, mem_alloc, cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_sum, d_sum, mem_alloc, cudaMemcpyDeviceToHost);

	cudaFree(d_in_1);
	cudaFree(d_out_1);
	cudaFree(d_raster1);
	cudaFree(d_in_2);
	cudaFree(d_out_2);
	cudaFree(d_raster2);
	cudaFree(d_affine_data);
	cudaFree(d_sum);

	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "write short took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
		<< " milliseconds\n";

	//trial_binary_render(h_sum, W, H);
	trial_binary_render(h_in_1, W, H);
	trial_binary_render(h_out_1, W, H);

	return 0;
}