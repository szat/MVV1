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

__global__ void cube(int * d_out, int * d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f*f*f;
}

int main(int argc, char ** argv) {
	const int numRows = 64;
	const int numCols = 32;
	const int ARRAY_SIZE = numRows * numCols;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	// generate the input array on the host
	int h_in[ARRAY_SIZE];
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			h_in[i*numCols + j] = (i) * (j) % 9;
		}
	}
	
	for (int i = 0; i < ARRAY_SIZE; i++) {
		if (i % numCols == 0) printf("\n");
		printf("%i ", h_in[i]);
	}
	printf("\n");

	int h_out[ARRAY_SIZE];

	// declare GPU memory pointers
	int * d_in;
	int * d_out;

	auto t1 = Clock::now();

	// allocate GPU memory
	cudaMalloc((void**)&d_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

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
