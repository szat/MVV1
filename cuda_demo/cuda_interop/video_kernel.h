#pragma once

void interpolate_frame(dim3 grid_size, dim3 block_size, uchar3* d_out_1, uchar3* d_out_2, uchar3* d_in_1, uchar3* d_in_2, uchar4* d_render_final,
	short* d_raster_1, short* d_raster_2, int w, int h, float * d_affine_data, int subdiv, float tau);
void reset_canvas(dim3 grid_size, dim3 block_size, uchar3* input, int w, int h);
void flip_image(dim3 grid_size, dim3 block_size, uchar4 *ptr, int w, int h);
void gaussian_2D_blur(dim3 grid_size, dim3 block_size, uchar4 *d_render_final, int w, int h, float *d_blur_coeff, int blur_radius);
