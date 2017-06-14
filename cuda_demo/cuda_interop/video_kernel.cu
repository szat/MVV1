#include "video_kernel.h"

__global__
void reset_image(uchar3* input, int w, int h) {
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int i = (row * w + col);

	// should not need to do this check if everything is good, must be an extra pixel
	if (i >= w * h) return;
	if ((row >= h) || (col >= w)) return;

	uchar3 blank = uchar3();
	blank.x = 0;
	blank.y = 0;
	blank.z = 0;
	input[i] = blank;
}

void my_cuda_func(dim3 block_size, dim3 grid_size, uchar3* input, int w, int h) {
	reset_image<< <block_size, grid_size>> >(input, w, h);
	cudaDeviceSynchronize();
}