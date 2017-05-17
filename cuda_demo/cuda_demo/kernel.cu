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
void kernel2D_subpix(uchar* d_output, uchar* d_input, int w, int h, float * d_affineData)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c;

	if ((r >= h) || (c >= w)) return;

	//going for subpixel accuracy
	int rowStep = 10;
	int colStep = 10;
	float dRow = 1 / rowStep;
	float dCol = 1 / colStep;
	int new_c;
	int new_r;

	for (int row = 0; row < rowStep; row++) {
		for (int col = 0; col < colStep; col++) {
			float sub_c = (float)c + dCol*(float)col;
			float sub_r = (float)r + dRow*(float)row;
			int new_c = (int)(d_affineData[0] * sub_c + d_affineData[1] * sub_r + d_affineData[4]);
			int new_r = (int)(d_affineData[2] * sub_c + d_affineData[3] * sub_r + d_affineData[5]);

			if ((new_r >= h) || (new_c >= w) || (new_r < 0) || (new_c < 0)) continue;

			int new_i = new_r * w + new_c;
			d_output[new_i] = d_input[i];
		}
	}
}

int main(int argc, char ** argv) {
	cout << "welcome to cuda_demo testing unit!" << endl;
	cout << "loading 2 images with openCV, processing and adding them with cuda (grayscale)." << endl;

	string address1 = "..\\data_store\\mona_lisa_1.jpg";
	Mat img1 = imread(address1, IMREAD_GRAYSCALE);
	cout << "loaded img1: " << address1 << endl;

	int W = img1.size().width;
	int H = img1.size().height;

	cout << "declaring host data-structures..." << endl;
	uchar *h_img1In;
	uchar *h_img1Out;
	
	h_img1In = (uchar*)malloc(W*H * sizeof(uchar));
	Mat img1Flat = img1.reshape(1, 1);
	h_img1In = img1Flat.data;

	h_img1Out = (uchar*)malloc(W*H * sizeof(uchar));
	for (int j = 0; j < W*H; j++) h_img1Out[j] = 0;

	float h_affineData[6] = {0.966, -0.259, 0, 0.259, 0.966, 0};

	//--Sending the data to the GPU memory
	cout << "declaring device data-structures..." << endl;

	float * d_affineData;
	cudaMalloc((void**)&d_affineData, 6 * sizeof(float));
	cudaMemcpy(d_affineData, h_affineData, 6 * sizeof(float), cudaMemcpyHostToDevice);

	uchar * d_img1In;
	cudaMalloc((void**)&d_img1In, W*H * sizeof(uchar));
	cudaMemcpy(d_img1In, h_img1In, W*H * sizeof(uchar), cudaMemcpyHostToDevice);

	uchar * d_img1Out;
	cudaMalloc((void**)&d_img1Out, W*H * sizeof(uchar));
	cudaMemcpy(d_img1Out, h_img1Out, W*H * sizeof(uchar), cudaMemcpyHostToDevice);

	//--GPU variables
	dim3 blockSize(32, 32);
	int bx = (W + 32 - 1) / 32;
	int by = (H + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);

	cout << "starting image transformation..." << endl;
	auto t1 = Clock::now();
	kernel2D<< <gridSize, blockSize >> >(d_img1Out, d_img1In, W, H, d_affineData);	
	auto t2 = Clock::now();
	std::cout << "delta time " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << std::endl;

	//--Send data back to the host from the GPU and free memory
	cudaMemcpy(h_img1Out, d_img1Out, W*H * sizeof(uchar), cudaMemcpyDeviceToHost);

	cudaFree(d_img1In);
	cudaFree(d_img1Out);
	cudaFree(d_affineData);

	Mat render1 = Mat(1, W*H, CV_8UC1, h_img1Out);
	render1 = render1.reshape(1, H);

	return 0;
}