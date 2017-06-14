#include "video_kernel.h"

__global__
void flip_y(uchar4 *ptr, int w, int h) {
	// map from threadIdx/BlockIdx to pixel position
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c;
	if ((r >= h) || (c >= w)) return;

	// only flip top
	if (r < h / 2) {
		int diff = h - r;
		int i_flip = diff * w + c;
		uchar4 temp = ptr[i_flip];
		ptr[i_flip] = ptr[i];
		ptr[i] = temp;
	}
}

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

void reset_canvas(dim3 block_size, dim3 grid_size, uchar3* input, int w, int h) {
	reset_image<< <block_size, grid_size>> >(input, w, h);
	cudaDeviceSynchronize();
}

void flip_image(dim3 block_size, dim3 grid_size, uchar4 *ptr, int w, int h) {
	flip_y << <block_size, grid_size >> > (ptr, w, h);
}