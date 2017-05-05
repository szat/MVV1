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
	const int numRows = 64;
	const int numCols = 32;
	const int ARRAY_SIZE = numRows * numCols;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	//Matrix in 2D indexing
	float matrix2D[numRows][numCols];
	//Populate
	printf("\nPopulate matrix2D(%d,%d) with double for loop...\n",numRows, numCols);
	for (int row = 0; row < numRows; row++) {
		for (int col = 0; col < numCols; col++) {
			matrix2D[row][col] = row + (float)col / 100;
		}
	}

	//Visualize canonically
	printf("\nVisualize matrix2D(%d,%d) with double for loop...\n", numRows, numCols);
	for (int row = 0; row < numRows; row++) {
		printf("\n");
		for (int col = 0; col < numCols; col++) {
			if (matrix2D[row][col] < 10) {
				printf(" %.2f ", matrix2D[row][col]);
			}
			else {
				printf("%.2f ", matrix2D[row][col]);
			}
		}
	}

	//Visualize matrix2D with 1D indexing, using unflatten()
	printf("\n\nVisualize matrix2D(%d,%d) with one for loop, using int * matIdx = unflatten(iter, %d, %d)...\n", numRows, numCols, numRows, numCols);
	for (int i = 0; i < numRows * numCols; i++) {
		if (i % numCols == 0) printf("\n");
		int * matIdx = unflatten_idx(i, numRows, numCols);
		float output = matrix2D[matIdx[0]][matIdx[1]];
		if ((int)output < 10) {
			printf(" %.2f ", output);
		}
		else {
			printf("%.2f ", output);
		}
	}

	//Matrix in 1D indexing
	float matrix1D[ARRAY_SIZE];
	//Populate
	printf("\n\nPopulate matrix1D(%d,%d) with double for loop, with matrix1D[row*numCols + col] (as used in flatten())...\n", numRows, numCols);
	for (int row = 0; row < numRows; row++) {
		for (int col = 0; col < numCols; col++) {
			matrix1D[row*numCols + col] = row + (float)col/100;
		}
	}

	//Visualize canonically in 1D
	printf("\nVisualize matrix1D(%d,%d) with one for loop...\n", numRows, numCols);
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i % numCols == 0) printf("\n");
		if ((int)matrix1D[i] < 10) {
			printf(" %.2f ", matrix1D[i]);
		}
		else {
			printf("%.2f ", matrix1D[i]);
		}
	}
	printf("\n");

	//Visualize canonically by construction in 2D
	printf("\nVisualize matrix1D(%d,%d) with double for loop, using construction indexing...\n", numRows, numCols, numRows, numCols);
	for (int i = 0; i < numRows; i++) {
		printf("\n");
		for (int j = 0; j < numCols; j++) {
			if (matrix1D[i*numCols + j] < 10) {
				printf(" %.2f ", matrix1D[i*numCols + j]);
			}
			else {
				printf("%.2f ", matrix1D[i*numCols + j]);
			}
		}
	}
	
	//Visualize matrix1D with 2D indexing, using flatten()
	printf("\n\nVisualize matrix1D(%d,%d) with double for loop, using flatten(iter*,%d,%d)...\n", numRows, numCols, numRows, numCols);
	for (int row = 0; row < numRows; row++) {
		printf("\n");
		for (int col = 0; col < numCols; col++) {
			int matrixIdx[] = { row, col };
			float output  = matrix1D[flatten_idx(matrixIdx, numRows, numCols)];
			if (output < 10) {
				printf(" %.2f ", output);
			}
			else {
				printf("%.2f ", output);
			}
		}
	}

	float h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d_in;
	float * d_out;

	auto t1 = Clock::now();

	// allocate GPU memory
	cudaMalloc((void**)&d_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, matrix1D, ARRAY_BYTES, cudaMemcpyHostToDevice);

	auto t2 = Clock::now();
	std::cout << "delta time " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000 << std::endl;

	// launch the kernel
	cube << <1, ARRAY_SIZE >> > (d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(d_in);
	cudaFree(d_out);

	std::cin.ignore();

	return 0;
}