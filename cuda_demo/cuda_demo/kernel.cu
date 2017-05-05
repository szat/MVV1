#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <conio.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
	printf("We are in thread %d\n", idx);
	//give a row per thread, in a 2d matrix it would be mat[idx][0], mat[idx][1], mat[idx][2], ...
	for (int col = 0; col < numCols; col++) {
		int idx2D[] = { idx, col };
		int matIdx = flatten_idx_gpu(idx2D, numRows, numCols);
		d_matrix1D_out[matIdx] = d_matrix1D_in[matIdx] + 1; //in our sample code should increment the rows from 2.14 to 3.14 for instance
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
	const int numRows = 32;
	const int numCols = 16;
	const int ARRAY_SIZE = numRows * numCols;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	//Matrix in 1D indexing
	float h_matrix1D_in[ARRAY_SIZE];

	//Populate
	printf("\n\nPopulate matrix1D(%d,%d) with double for loop, with matrix1D[row*numCols + col] (as used in flatten())...\n", numRows, numCols);
	for (int row = 0; row < numRows; row++) {
		for (int col = 0; col < numCols; col++) {
			h_matrix1D_in[row*numCols + col] = row + (float)col/100;
		}
	}

	//Visualize canonically in 1D
	/*
	printf("\nVisualize matrix1D_in(%d,%d) with one for loop...\n", numRows, numCols);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i % numCols == 0) printf("\n");
		if ((int)h_matrix1D_in[i] < 10) {
			printf(" %.2f ", h_matrix1D_in[i]);
		}
		else {
			printf("%.2f ", h_matrix1D_in[i]);
		}
	}
	printf("\n");
	*/

	float h_matrix1D_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_matrix1D_in;
	float * d_matrix1D_out;

	

	// allocate GPU memory
	cudaMalloc((void**)&d_matrix1D_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_matrix1D_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_matrix1D_in, h_matrix1D_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	auto t1 = Clock::now();
	auto t2 = Clock::now();
	std::cout << "delta time " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000 << std::endl;

	// launch the kernel
	// rowOperation << <1, numRows >> > (d_matrix1D_out, d_matrix1D_in, numRows, numCols);
	// colOperation << <1, numCols >> > (d_matrix1D_out, d_matrix1D_in, numRows, numCols);
	// rowOperation_block << <numRows, numCols >> > (d_matrix1D_out, d_matrix1D_in, numRows, numCols);
	colOperation_block << < numCols, numRows >> > (d_matrix1D_out, d_matrix1D_in, numRows, numCols);
	// cube << <1, ARRAY_SIZE >> > (d_matrix1D_out, d_matrix1D_in);
	
	// copy back the result array to the CPU
	cudaMemcpy(h_matrix1D_out, d_matrix1D_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	//Visualize canonically in 1D
	printf("\nVisualize matrix1D_in(%d,%d) with one for loop...\n", numRows, numCols);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i % numCols == 0) printf("\n");
		if ((int)h_matrix1D_out[i] < 10) {
			printf(" %.2f ", h_matrix1D_out[i]);
		}
		else {
			printf("%.2f ", h_matrix1D_out[i]);
		}
	}
	printf("\n");


	cudaFree(d_matrix1D_in);
	cudaFree(d_matrix1D_out);

	std::cin.ignore();

	return 0;
}