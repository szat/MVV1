#include <stdio.h>
#include <iostream>
<<<<<<< HEAD
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


//CUDA demo 3
/*
=======
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
>>>>>>> 0f000086da415b4307c4a514d2a4c957859726b7
#include <stdio.h>
#include "gputimer.h"

#define NUM_THREADS 10000000
#define ARRAY_SIZE  100

#define BLOCK_WIDTH 1000

<<<<<<< HEAD
void print_array(int *array, int size)
{
	printf("{ ");
	for (int i = 0; i < size; i++) { printf("%d ", array[i]); }
	printf("}\n");
}

__global__ void increment_naive(int *g)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE;
	g[i] = g[i] + 1;
}

__global__ void increment_atomic(int *g)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % ARRAY_SIZE;
	atomicAdd(&g[i], 1);
}

int main(int argc, char **argv)
{
	GpuTimer timer;
	printf("%d total threads in %d blocks writing into %d array elements\n",
		NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

	// declare and allocate host memory
	int *h_array = new int[ARRAY_SIZE]; //forum suggestion against stack overflow
	//int h_array[ARRAY_SIZE]; //original code
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	// declare, allocate, and zero out GPU memory
	int * d_array;
	cudaMalloc((void **)&d_array, ARRAY_BYTES);
	cudaMemset((void *)d_array, 0, ARRAY_BYTES);

	// launch the kernel - comment out one of these
	timer.Start();

	// Instructions: This program is needed for the next quiz
	// uncomment increment_naive to measure speed and accuracy 
	// of non-atomic increments or uncomment increment_atomic to
	// measure speed and accuracy of  atomic icrements
	// increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
	increment_atomic << <NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH >> >(d_array);
	timer.Stop();

	// copy back the array of sums from GPU and print
	cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	print_array(h_array, ARRAY_SIZE);
	printf("Time elapsed = %g ms\n", timer.Elapsed());

	// free GPU memory allocation and exit
	cudaFree(d_array);
	std::cin.ignore();
	return 0;
}
*/

/*
//CUDA demo 2
#define NUM_BLOCKS 16
#define BLOCK_WIDTH 1

__global__ void hello()
{
	printf("Hello world! I'm a thread in block %d\n", blockIdx.x);
}


int main(int argc, char **argv)
{
	// launch the kernel
	hello << <NUM_BLOCKS, BLOCK_WIDTH >> >();

	// force the printf()s to flush
	cudaDeviceSynchronize();

	printf("That's all!\n");

	std::cin.ignore();
	return 0;
}
*/

//CUDA demo 1
/*
__global__ void cube(float * d_out, float * d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f*f*f;
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 96;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
=======
__global__ void cube(float *d_out, float *d_in) {
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f * f * f;
}

int main(int argc, char **argv) {
	const int ARRAY_SIZE = 1024;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

>>>>>>> 0f000086da415b4307c4a514d2a4c957859726b7
	float h_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h_in[i] = float(i);
	}
	float h_out[ARRAY_SIZE];

<<<<<<< HEAD
	// declare GPU memory pointers
=======
	// gpu memory pointers
>>>>>>> 0f000086da415b4307c4a514d2a4c957859726b7
	float * d_in;
	float * d_out;

	// allocate GPU memory
<<<<<<< HEAD
	cudaMalloc((void**)&d_in, ARRAY_BYTES);
	cudaMalloc((void**)&d_out, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	cube << <1, ARRAY_SIZE >> > (d_out, d_in);

	// copy back the result array to the CPU
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
=======
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **)&d_out, ARRAY_BYTES);

	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cube<<<1, ARRAY_SIZE>>>(d_out, d_in);
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

>>>>>>> 0f000086da415b4307c4a514d2a4c957859726b7
	for (int i = 0; i < ARRAY_SIZE; i++) {
		printf("%f", h_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

<<<<<<< HEAD
=======
	// deallocate GPU memory
>>>>>>> 0f000086da415b4307c4a514d2a4c957859726b7
	cudaFree(d_in);
	cudaFree(d_out);

	std::cin.ignore();

	return 0;
<<<<<<< HEAD
}
*/
=======
}
>>>>>>> 0f000086da415b4307c4a514d2a4c957859726b7
