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
void kernel2D(uchar* d_output, uchar* d_input, int w, int h, float tau)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c; 
	
	if ((r >= h) || (c >= w)) return;

	d_output[i] = d_input[i] / 2;
}

__device__ 
void affine_push_pt_gpu(float * newCoord, float * oldCoord, float * affineData, float tau) {
	/*
	Parametrization should go:
	
	A = [	(1-t) + a_00 * t, a_01 * t,
			a_10 * t		, (1-t) + a_11 * t]
	
	B = [t * b_0, t * b_1]

	((1-t) + a_00 * t)x + (a_01 * t)y + t * b_0, (a_10 * t)x + ((1-t) + a_11 * t)y) + t * b_1 
	*/

	float cNew = ((1 - tau) + affineData[0] * tau) * oldCoord[0] + (affineData[1] * tau) * oldCoord[1] + tau*affineData[4];
	float rNew = (affineData[2] * tau) * oldCoord[0] + ((1 - tau) + affineData[3] * tau) * oldCoord[1] + tau*affineData[5];
	
	//float cNew = oldCoord[0];
	//float rNew = oldCoord[1];
	newCoord[0] = cNew;
	newCoord[1] = rNew;
}

__global__
void push_image_gpu(uchar* d_output, uchar* d_input, int w, int h, float tau, short* d_triangleRaster, float * d_triangleData) {
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = r * w + c;

	if ((r >= h) || (c >= w)) return;

	int triangleIdx = d_triangleRaster[idx];
	
	if (triangleIdx < 0) return;

	float affineData[6] = {
		d_triangleData[triangleIdx],		d_triangleData[triangleIdx + 1],
		d_triangleData[triangleIdx + 2],	d_triangleData[triangleIdx + 3],
		d_triangleData[triangleIdx + 4],	d_triangleData[triangleIdx + 5]
	};

	float oldCoord[2] = { c, r };	
	float newCoord[2];

	//--The computation part
	affine_push_pt_gpu(newCoord, oldCoord, affineData, tau);
	
	if (((int)newCoord[1] >= h) || ((int)newCoord[0] >= w) || (int)newCoord[1] < 0 || (int)newCoord[0] < 0) return;

	//--Snapping the new coordinate to the pixel
	int newIdx = newCoord[1] * w + newCoord[0]; //newC = newCoord[0], newR = newCoord[1];

	d_output[newIdx] = d_input[idx] * tau;
}

int main(int argc, char ** argv) {
	cout << "Welcome to cuda_demo testing unit!" << endl;
	cout << "Loading one image with openCV! (grayscale)" << endl;

	string address1 = "..\\data_store\\medium_picture.jpg";
	Mat img1 = imread(address1, IMREAD_GRAYSCALE);

	int triangleNb = 1;
	int W = img1.size().width;
	int H = img1.size().height;

	uchar *h_imgIn;
	uchar *h_imgOut;
	
	h_imgIn = (uchar*)malloc(W*H * sizeof(uchar));
	Mat img1Flat = img1.reshape(1, 1);
	h_imgIn = img1Flat.data;

	h_imgOut = (uchar*)malloc(W*H * sizeof(uchar));
	for (int j = 0; j < W*H; j++) h_imgOut[j] = 0;

	//--Dummy values, to be replaced later by the actual raster
	short * h_triangleRaster = new short[W*H];
	for (int j = 0; j < W*H; j++) h_triangleRaster[j] = 0;

	float h_triangleData[6] = { 1, 0.2, 0, 1, 50, 50}; //corresponding to the null transformation

	//--Sending the data to the GPU memory
	short * d_triangleRaster;
	cudaMalloc((void**)&d_triangleRaster, W*H * sizeof(short));
	cudaMemcpy(d_triangleRaster, h_triangleRaster, W*H * sizeof(short), cudaMemcpyHostToDevice);

	float * d_triangleData;
	cudaMalloc((void**)&d_triangleData, 6 * sizeof(float));
	cudaMemcpy(d_triangleData, h_triangleData, 6 * sizeof(float), cudaMemcpyHostToDevice);

	uchar * d_imgIn;
	cudaMalloc((void**)&d_imgIn, W*H * sizeof(uchar));
	cudaMemcpy(d_imgIn, h_imgIn, W*H * sizeof(uchar), cudaMemcpyHostToDevice);

	uchar * d_imgOut;
	cudaMalloc((void**)&d_imgOut, W*H * sizeof(uchar));
	cudaMemcpy(d_imgOut, h_imgOut, W*H * sizeof(uchar), cudaMemcpyHostToDevice);

	//--GPU variables

	dim3 blockSize(32, 32);
	int bx = (W + 32 - 1) / 32;
	int by = (H + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);

	auto t1 = Clock::now();
	push_image_gpu << <gridSize, blockSize >> >(d_imgOut, d_imgIn, W, H, 1, d_triangleRaster, d_triangleData);	
	auto t2 = Clock::now();
	std::cout << "delta time " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << std::endl;

	//--Send data back to the host from the GPU and free memory
	cudaMemcpy(h_imgOut, d_imgOut, W*H * sizeof(uchar), cudaMemcpyDeviceToHost);

	cudaFree(d_imgIn);
	cudaFree(d_imgOut);
	cudaFree(d_triangleData);
	cudaFree(h_triangleRaster);

	Mat render1 = Mat(1, W*H, CV_8UC1, h_imgOut);
	render1 = render1.reshape(1, H);

	return 0;
}