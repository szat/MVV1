#include "video_kernel.h"

__global__
void kernel2D_subpix(uchar3* d_output, uchar3* d_input, short* d_raster1, int w, int h, float * d_affineData, int subDiv, float tau, bool reverse)
{
	if (tau > 1 || tau < 0) return;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int raster_index = (row * w + col);
	//int color_index = raster_index * 3;


	short affine_index = d_raster1[raster_index];
	short offset = (affine_index - 1) * 12;
	if (reverse) {
		offset += 6;
	}
	if (affine_index != 0) {
		// triangle indexes start at 1
		float diff = 1 / (float)subDiv;
		for (int i = 0; i < subDiv; i++) {
			for (int j = 0; j < subDiv; j++) {
				int new_c = (int)(((1 - tau) + tau*d_affineData[offset]) * (float)(col - 0.5 + (diff * i)) + (tau * d_affineData[offset + 1]) * (float)(row - 0.5 + (diff * j)) + (tau * d_affineData[offset + 2]));
				int new_r = (int)((tau * d_affineData[offset + 3]) * (float)(col - 0.5 + (diff * i)) + ((1 - tau) + tau * d_affineData[offset + 4]) * (float)(row - 0.5 + (diff * j)) + (tau * d_affineData[offset + 5]));
				if ((new_r >= h) || (new_c >= w) || (new_r < 0) || (new_c < 0)) {
					return;
				}
				int new_i = new_r * w + new_c;
				d_output[new_i] = d_input[raster_index];
			}
		}
	}
}

// Conversion functions that are currently unused, but might be used in the future.
/*
__global__
void convert_uchar3_to_uchar4(uchar3 *input, uchar4 *output, int w, int h) {
int col = blockIdx.x*blockDim.x + threadIdx.x;
int row = blockIdx.y*blockDim.y + threadIdx.y;
int index = (row * w + col);

if (index >= w * h) return;
if ((row >= h) || (col >= w)) return;

uchar4 new_uchar4 = uchar4();
new_uchar4.x = input[index].x;
new_uchar4.y = input[index].y;
new_uchar4.z = input[index].z;
output[index] = new_uchar4;
}

__global__
void convert_uchar4_to_uchar3(uchar4 *input, uchar3 *output, int w, int h) {
int col = blockIdx.x*blockDim.x + threadIdx.x;
int row = blockIdx.y*blockDim.y + threadIdx.y;
int index = (row * w + col);

if (index >= w * h) return;
if ((row >= h) || (col >= w)) return;

uchar3 new_uchar3 = uchar3();
new_uchar3.x = input[index].x;
new_uchar3.y = input[index].y;
new_uchar3.z = input[index].z;
output[index] = new_uchar3;
}
*/

__global__
void kernel2D_add(uchar4* d_output, uchar3* d_input_1, uchar3* d_input_2, int w, int h, float tau) {
	// I am also sorting out the color channel issues in this function.

	//tau is from a to b
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int raster_index = (row * w + col);

	// should not need to do this check if everything is good, must be an extra pixel
	if (raster_index >= w * h) return;
	if ((row >= h) || (col >= w)) return;

	if (d_input_1[raster_index].x == 0 && d_input_1[raster_index].y == 0 && d_input_1[raster_index].z == 0) {
		uchar4 new_uchar4 = uchar4();
		new_uchar4.x = d_input_2[raster_index].z;
		new_uchar4.y = d_input_2[raster_index].y;
		new_uchar4.z = d_input_2[raster_index].x;
		d_output[raster_index] = new_uchar4;
	}
	else if (d_input_2[raster_index].x == 0 && d_input_2[raster_index].y == 0 && d_input_2[raster_index].z == 0) {
		uchar4 new_uchar4 = uchar4();
		new_uchar4.x = d_input_1[raster_index].z;
		new_uchar4.y = d_input_1[raster_index].y;
		new_uchar4.z = d_input_1[raster_index].x;
		d_output[raster_index] = new_uchar4;
	}
	else {
		d_output[raster_index].x = tau*d_input_1[raster_index].z + (1 - tau)*d_input_2[raster_index].z;
		d_output[raster_index].y = tau*d_input_1[raster_index].y + (1 - tau)*d_input_2[raster_index].y;
		d_output[raster_index].z = tau*d_input_1[raster_index].x + (1 - tau)*d_input_2[raster_index].x;
	}
}


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

void interpolate_frame(dim3 grid_size, dim3 block_size, uchar3* d_out_1, uchar3* d_out_2, uchar3* d_in_1, uchar3* d_in_2, uchar4* d_render_final, 
	short* d_raster_1, short* d_raster_2, int w, int h, float * d_affine_data, int subdiv, float tau) {

	float reverse_tau = 1.0f - tau;
	kernel2D_subpix << <grid_size, block_size >> >(d_out_1, d_in_1, d_raster_1, w, h, d_affine_data, subdiv, tau, false);
	//kernel2D_subpix << <grid_size, block_size >> >(d_out_2, d_in_2, d_raster_2, w, h, d_affine_data, subdiv, reverse_tau, true);
	kernel2D_add << <grid_size, block_size >> > (d_render_final, d_out_1, d_out_2, w, h, tau);
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