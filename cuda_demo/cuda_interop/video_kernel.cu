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
void gaussian_blur(uchar4 *d_render_final, int w, int h, float *d_blur_coeff, int blur_radius, bool vertical) {
	// map from threadIdx/BlockIdx to pixel position
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int index = r * w + c;
	if ((r >= h) || (c >= w)) return;

	float gaussian_r = 0;
	float gaussian_g = 0;
	float gaussian_b = 0;

	int min = -1 * blur_radius;
	int max = blur_radius;

	int box_width = 2 * blur_radius + 1;
	int new_index = 0;

	for (int i = min; i <= max; i++) {
		// switch depending on the direction of the gaussian blur
		if (vertical) {
			new_index = index + i * w;
		}
		else {
			new_index = index + i;
		}

		int coeff_index = i + blur_radius;
		float coeff = d_blur_coeff[coeff_index];

		gaussian_r = gaussian_r + coeff * d_render_final[new_index].x;
		gaussian_g = gaussian_g + coeff * d_render_final[new_index].y;
		gaussian_b = gaussian_b + coeff * d_render_final[new_index].z;

		// this will cause light backgrounds to darken
		// TODO: Fix this bug
	}

	// sync threads
	__syncthreads();

	uchar4 result = uchar4();
	result.x = gaussian_r;
	result.y = gaussian_g;
	result.z = gaussian_b;
	d_render_final[index] = result;
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

void gaussian_2D_blur(dim3 block_size, dim3 grid_size, uchar4 *d_render_final, int w, int h, float *d_blur_coeff, int blur_radius) {
	// 2d gaussian blur can be decoupled into X and Y
	// horizontal blur first
	gaussian_blur << < grid_size, block_size >> > (d_render_final, w, h, d_blur_coeff, blur_radius, false);
	// vertical blur next
	gaussian_blur << < grid_size, block_size >> > (d_render_final, w, h, d_blur_coeff, blur_radius, true);
}

void flip_image(dim3 block_size, dim3 grid_size, uchar4 *ptr, int w, int h) {
	flip_y << <block_size, grid_size >> > (ptr, w, h);
}