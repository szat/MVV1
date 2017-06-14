#pragma once

void reset_canvas(dim3 block_size, dim3 grid_size, uchar3* input, int w, int h);
void flip_image(dim3 block_size, dim3 grid_size, uchar4 *ptr, int w, int h);
void gaussian_2D_blur(dim3 block_size, dim3 grid_size, uchar4 *d_render_final, int w, int h, float *d_blur_coeff, int blur_radius);
