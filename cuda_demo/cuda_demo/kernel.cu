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

//load 2d array
//do something on 2d array
//send back the 2d array

__global__ void image_to_inter_1C(unsigned char * d_matrix1D_out, unsigned char * d_matrix1D_in, int numRows, int numCols, float tau) {
	//int rowIdx = blockIdx.x; //give a row per block ==> numRows
	//int colIdx = threadIdx.x; //give a col per thread, thus a pixel per thread ==> numCols
	//if (rowIdx < numRows && colIdx < numCols && 0 <= tau && tau <= 1) {
		//int idx2D_in[] = { rowIdx, colIdx };
		//int matIdx_in = flatten_idx_gpu(idx2D_in, numRows, numCols);
		//d_matrix1D_out[matIdx_in] = d_matrix1D_in[matIdx_in];
		//d_matrix1D_out[matIdx_in] = (unsigned char) (tau * (short) d_matrix1D_in[matIdx_in]);
		/*
		int matIdx_in = flatten_idx_gpu(idx2D_in, numRows, numCols);
		int triangleIdx = d_matrixTriangles1D_in[matIdx_in];
		if (triangleIdx != -1) {
			float triangleData[6];
			for (int i = 0; i < 6; i++) triangleData[i] = d_triangleData1D_in[triangleIdx + i];
			//using triangleData project pixel in d_matrix1D_in at position matIdx_in onto pixel in d_matrix1D_out at position matIdx_out
			int idx2D_out[] = { (int)rowIdx*tau, (int)colIdx*tau };
			int matIdx_out = flatten_idx_gpu(idx2D_out, numRows, numCols);
			d_matrix1D_out[matIdx_out] = d_matrix1D_in[matIdx_in];
		}
		*/
	//}
}

__global__
void kernel2D(uchar* d_output, uchar* d_input, int w, int h, float tau)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c; 
	
	if ((r >= h) || (c >= w)) return;

	d_output[i] = d_input[i] / 2;
	/*
	d_output[i].x = (uchar) tau * d_input[i].x;    //Compute red
	d_output[i].y = (uchar) tau * d_input[i].y; //Compute green
	d_output[i].z = (uchar) tau * d_input[i].z;  //Compute blue
	d_output[i].w = 255; // Fully 
	*/
}

__device__ 
void affine_transform(float * newCoord, float * oldCoord, float * affineData) {
	float cNew = oldCoord[0];
	float rNew = oldCoord[1];
	newCoord[0] = cNew;
	newCoord[1] = rNew;
}

__global__
void raster2D(uchar* d_output, uchar* d_input, int w, int h, float tau, short* d_triangleRaster, float * d_triangleData) {
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c;

	if ((r >= h) || (c >= w)) return;

	int triangleIdx = d_triangleRaster[i];

	float affineData[6] = {
		d_triangleData[triangleIdx],		d_triangleData[triangleIdx + 1],
		d_triangleData[triangleIdx + 2],	d_triangleData[triangleIdx + 3],
		d_triangleData[triangleIdx + 4],	d_triangleData[triangleIdx + 5]
	};

	//do operations on c and r (col and row)
	float oldCoord[2] = { c, r };
	float newCoord[2];

	affine_transform(newCoord, oldCoord, affineData);
	
	int newC = (int) newCoord[0];
	int newR = (int) newCoord[1];
	int newI = newR * w + newC;

	d_output[newI] = d_input[i] /5;
}

int main(int argc, char ** argv) {
	cout << "Welcome to cuda_demo testing unit!" << endl;
	cout << "Loading one image with openCV! (grayscale)" << endl;

	string address1 = "..\\data_store\\big_picture.jpg";
	Mat img1 = imread(address1, IMREAD_GRAYSCALE);

	int triangleNb = 1;
	int W = img1.size().width;
	int H = img1.size().height;

	uchar *h_in;
	uchar *h_out;
	
	h_in = (uchar*)malloc(W*H * sizeof(uchar));
	h_out = (uchar*)malloc(W*H * sizeof(uchar));

	Mat img1Flat = img1.reshape(1, 1);
	h_in = img1Flat.data;

	uchar * d_in;
	uchar * d_out;
	cudaMalloc((void**)&d_in, W*H * sizeof(uchar));
	cudaMalloc((void**)&d_out, W*H * sizeof(uchar));

	cudaMemcpy(d_in, h_in, W*H * sizeof(uchar), cudaMemcpyHostToDevice);
	
	short * h_triangleRaster = new short[W*H];
	for (int i = 0; i < W*H; i++) h_triangleRaster[i] = 0;
	float h_triangleData[6] = { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };

	short * d_triangleRaster;
	float * d_triangleData;

	cudaMalloc((void**)&d_triangleRaster, W*H * sizeof(short));
	cudaMemcpy(d_triangleRaster, h_triangleRaster, W*H * sizeof(short), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_triangleData, 6 * sizeof(float));
	cudaMemcpy(d_triangleData, h_triangleData, 6 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockSize(32, 32);

	int bx = (W + 32 - 1) / 32;
	int by = (H + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);



	auto t1 = Clock::now();
	raster2D << <gridSize, blockSize >> >(d_out, d_in, W, H, 0.3, d_triangleRaster, d_triangleData);	
	auto t2 = Clock::now();
	std::cout << "delta time " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << std::endl;

	cudaMemcpy(h_out, d_out, W*H * sizeof(uchar), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	Mat out = Mat(1, W*H, CV_8UC1, h_out);
	out = out.reshape(1, H);

	return 0;
}