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

__global__ void cube(float * d_out, float * d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f*f*f;
}

__device__ int flatten_idx_gpu(int * matrixIdx, int numRows, int numCols) {
	if (0 <= matrixIdx[0] < numRows && 0 <= matrixIdx[1] < numCols) return matrixIdx[0] * numCols + matrixIdx[1];
	else return 0;
}

__device__ int * unflatten_idx_gpu(int arrayIdx, int numRows, int numCols) {
	if (0 <= arrayIdx < numRows*numCols) {
		int rowIdx = arrayIdx / numCols;
		int idx[] = { rowIdx ,  arrayIdx - rowIdx*numCols };
		return idx;
	}
	return 0;
}

__global__ void rowOperation_block(float * d_matrix1D_out, float * d_matrix1D_in, int numRows, int numCols) {
	//number of blocks should be the number of rows
	//number of threads should be the number of cols
	int idxBlk = blockIdx.x;
	int idxThd = threadIdx.x;
	//give a row per block, in a 2d matrix it would be mat[idxBlk][idxThd], mat[idxBlk][idxThd], mat[idxBlk][idxThd], ...
	int idx2D[] = { idxBlk, idxThd };
	int matIdx = flatten_idx_gpu(idx2D, numRows, numCols);
	d_matrix1D_out[matIdx] = d_matrix1D_in[matIdx] + 1;
}

__global__ void colOperation_block(float * d_matrix1D_out, float * d_matrix1D_in, int numRows, int numCols) {
	int idxBlk = blockIdx.x;
	int idxThd = threadIdx.x;
	//give a column per thread, in a 2d matrix it would be mat[idxThd][idxBlk], mat[idxThd][idxBlk], mat[idxThd][idxBlk], ...
	int idx2D[] = { idxThd, idxBlk };
	int matIdx = flatten_idx_gpu(idx2D, numRows, numCols);
	d_matrix1D_out[matIdx] = d_matrix1D_in[matIdx] + 0.01;
}

__global__ void rowOperation(float * d_matrix1D_out, float * d_matrix1D_in, int numRows, int numCols) {
	int idx = threadIdx.x;
	//printf("We are in thread %d\n", idx);
	//give a row per thread, in a 2d matrix it would be mat[idx][0], mat[idx][1], mat[idx][2], ...
	for (int col = 0; col < numCols; col++) {
		int idx2D[] = { idx, col };
		int matIdx = flatten_idx_gpu(idx2D, numRows, numCols);
		int matIdx_out = 3 * (matIdx + 32) % numCols;
		d_matrix1D_out[matIdx_out] = d_matrix1D_in[matIdx] + 1; //in our sample code should increment the rows from 2.14 to 3.14 for instance
	}
}

__global__ void image_to_inter_1C(unsigned char * d_matrix1D_out, unsigned char * d_matrix1D_in, int numRows, int numCols, float tau) {
	int rowIdx = blockIdx.x; //give a row per block ==> numRows
	int colIdx = threadIdx.x; //give a col per thread, thus a pixel per thread ==> numCols
	if (rowIdx < numRows && colIdx < numCols && 0 <= tau && tau <= 1) {
		int idx2D_in[] = { rowIdx, colIdx };
		int matIdx_in = flatten_idx_gpu(idx2D_in, numRows, numCols);
		d_matrix1D_out[matIdx_in] = (unsigned char) (tau * (short) d_matrix1D_in[matIdx_in]);
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
	}
}

__global__ void colOperation(float * d_matrix1D_out, float * d_matrix1D_in, int numRows, int numCols) {
	int idx = threadIdx.x;
	//give a column per thread, in a 2d matrix it would be mat[0][idx], mat[1][idx], mat[2][idx], ...
	for (int row = 0; row < numRows; row++) {
		int idx2D[] = { row, idx };
		int matIdx = flatten_idx_gpu(idx2D, numRows, numCols);
		d_matrix1D_out[matIdx] = d_matrix1D_in[matIdx] + 0.01; //in ou sample code should increment the cols from 2.14 to 2.15
	}
}

int flatten_idx(int * matrixIdx, int numRows, int numCols) {
	if (0 <= matrixIdx[0] < numRows && 0 <= matrixIdx[1] < numCols) return matrixIdx[0] * numCols + matrixIdx[1];
	else return 0;
}

int * unflatten_idx(int arrayIdx, int numRows, int numCols) {
	if (0 <= arrayIdx < numRows*numCols) {
		int rowIdx = arrayIdx / numCols;
		int idx[] = { rowIdx ,  arrayIdx - rowIdx*numCols };
		return idx;
	}
	return 0;
}

int main(int argc, char ** argv) {
	cout << "Welcome to cuda_demo testing unit!" << endl;
	cout << "Loading one image with openCV! (grayscale)" << endl;

	string address1 = "..\\data_store\\medium_picture.jpg";
	Mat img1 = imread(address1, IMREAD_GRAYSCALE);

	const int numRows = img1.rows;
	const int numCols = img1.cols;

	const int ARRAY_SIZE = numRows * numCols;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned char);

	unsigned char *h_img1Data_in;
	unsigned char *h_img1Data_out;
	h_img1Data_out = (unsigned char *)malloc(ARRAY_BYTES);
	h_img1Data_in = (unsigned char *)malloc(ARRAY_BYTES);
	
	
	Mat img1Flat = img1.reshape(1, 1);
	h_img1Data_in = img1Flat.data;

	//for(int i = 0; i < ARRAY_SIZE; i++) h_img1Data_out[i] = 0; //just to put something in it

	unsigned char * d_img1Data_in;
	unsigned char * d_img1Data_out;

	

	cudaMalloc((void**)&d_img1Data_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_img1Data_out, ARRAY_BYTES);


	for (int i = 0; i < 30; i++) {
		
		
		cudaMemcpy(d_img1Data_in, h_img1Data_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
		auto t1 = Clock::now();
		image_to_inter_1C << <numRows, numCols >> > (d_img1Data_out, d_img1Data_in, numRows, numCols, 0.7);
		
		cudaMemcpy(h_img1Data_out, d_img1Data_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
		auto t2 = Clock::now();
		
		std::cout << "delta time " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << std::endl;
	}

	Mat out = Mat(1, numRows*numCols, CV_8UC1, h_img1Data_out);

	cout << "out size height " << out.size().height << endl;
	cout << "out size width " << out.size().width << endl;

	out = out.reshape(1, numRows);

	cout << "out size height " << out.size().height << endl;
	cout << "out size width " << out.size().width << endl;



	namedWindow("Result", WINDOW_AUTOSIZE);
	imshow("Result", img1);
	waitKey(0);
	cin.ignore();

	//cudaThreadSynchronize();

	//Free
	cudaFree(d_img1Data_in);
	cudaFree(d_img1Data_out);

	

	return 0;
}