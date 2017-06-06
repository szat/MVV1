// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include "binary_read.h"

using namespace std;

#define REFRESH_DELAY     2 //ms

//TRY TO CALL GLUTPOSTREDISPLAY FROM A FOOR LOOP
GLuint  bufferObj;

cudaGraphicsResource *resource;
__device__ int counter;
__device__ volatile int param = 50;


__global__
void kernel2D_subpix(uchar3* d_output, uchar3* d_input, short* d_raster1, int *d_error_tracker, int w, int h, float * d_affineData, int subDiv, float tau, bool reverse)
{
	if (tau > 1 || tau < 0) return;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int raster_index = (row * w + col);
	//int color_index = raster_index * 3;

	// should not need to do this check if everything is good, must be an extra pixel
	if (raster_index >= w * h) { 
		d_error_tracker[0]++;
		return;
	}
	if ((row >= h) || (col >= w)) {
		d_error_tracker[1]++;
		return;
	}

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
					d_error_tracker[2]++;
					return;
				}
				int new_i = new_r * w + new_c;
				d_output[new_i] = d_input[raster_index];
			}
		}
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

__global__
void kernel2D_add(uchar4* d_output, uchar3* d_input_1, uchar3* d_input_2, int w, int h, float tau) {
	//tau is from a to b
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int raster_index = (row * w + col);

	// should not need to do this check if everything is good, must be an extra pixel
	if (raster_index >= w * h) return;
	if ((row >= h) || (col >= w)) return;

	if (d_input_1[raster_index].x == 0 && d_input_1[raster_index].y == 0 && d_input_1[raster_index].z == 0) {
		uchar4 new_uchar4 = uchar4();
		new_uchar4.x = d_input_2[raster_index].x;
		new_uchar4.y = d_input_2[raster_index].y;
		new_uchar4.z = d_input_2[raster_index].z;
		d_output[raster_index] = new_uchar4;
	}
	else if (d_input_2[raster_index].x == 0 && d_input_2[raster_index].y == 0 && d_input_2[raster_index].z == 0) {
		uchar4 new_uchar4 = uchar4();
		new_uchar4.x = d_input_1[raster_index].x;
		new_uchar4.y = d_input_1[raster_index].y;
		new_uchar4.z = d_input_1[raster_index].z;
		d_output[raster_index] = new_uchar4;
	}
	else {
		d_output[raster_index].x = tau*d_input_1[raster_index].x + (1 - tau)*d_input_2[raster_index].x;
		d_output[raster_index].y = tau*d_input_1[raster_index].y + (1 - tau)*d_input_2[raster_index].y;
		d_output[raster_index].z = tau*d_input_1[raster_index].z + (1 - tau)*d_input_2[raster_index].z;
	}
}

__global__
void gaussian_blur(uchar4 *d_render_final, int w, int h, float *d_blur_coeff, int blur_radius) {
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

	for (int i = 0; i <= box_width; i++) {
		for (int j = 0; j < box_width; j++) {
			// get new index
			int i_offset = min + i;
			int j_offset = min + j;
			int new_index = index + i_offset * w + j_offset;
			// check if out of bounds
			if (new_index > 0 && new_index < w * h) {
				// if inside bounds, add to total 
				float coeff = 0.04;
				gaussian_r = gaussian_r + coeff * d_render_final[new_index].x;
				gaussian_g = gaussian_r + coeff * d_render_final[new_index].y;
				gaussian_b = gaussian_r + coeff * d_render_final[new_index].z;
			}
			// this will cause light backgrounds to darken
			// TODO: Fix this bug
		}
	}

	// sync threads
	__syncthreads();

	uchar4 result = uchar4();
	result.x = gaussian_r;
	result.y = gaussian_g;
	result.z = gaussian_b;

	d_render_final[index] = result;
}


__global__ void flip_y(uchar4 *ptr, int w, int h) {
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

static void key_func(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		// clean up OpenGL and CUDA
		cudaGraphicsUnregisterResource(resource);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
		exit(0);
	}
}

static void draw_func(void) {
	// we pass zero as the last parameter, because out bufferObj is now
	// the source, and the field switches from being a pointer to a
	// bitmap to now mean an offset into a bitmap object

	int width = 667;
	int height = 1000;

	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

float * calculate_blur_coefficients(int blur_radius) {
	/*
	time for some normalization
	normalization constraints for this gaussian:
	
	- the sum of the (2r+1)^2 gaussian 
	- for now i've just hard-coded a pseudo-gaussian
	*/ 
	
	// I DON'T EVEN CARE HOW DUMB THIS IS TO HARD-CODE IT
	// I'M TAKING THIS OUT
	// TEMPORARY

	float *test_coefficients = new float[(blur_radius * 2 + 1) * (blur_radius * 2 + 1)];
	test_coefficients[0] = 0.01;
	test_coefficients[1] = 0.025;
	test_coefficients[2] = 0.05;
	test_coefficients[3] = 0.025;
	test_coefficients[4] = 0.01;
	test_coefficients[5] = 0.025;
	test_coefficients[6] = 0.05;
	test_coefficients[7] = 0.07;
	test_coefficients[8] = 0.05;
	test_coefficients[9] = 0.025;
	test_coefficients[10] = 0.05;
	test_coefficients[11] = 0.07;
	test_coefficients[12] = 0.08;
	test_coefficients[13] = 0.07;
	test_coefficients[14] = 0.05;
	test_coefficients[15] = 0.025;
	test_coefficients[16] = 0.05;
	test_coefficients[17] = 0.07;
	test_coefficients[18] = 0.05;
	test_coefficients[19] = 0.025;
	test_coefficients[20] = 0.01;
	test_coefficients[21] = 0.025;
	test_coefficients[22] = 0.05;
	test_coefficients[23] = 0.025;
	test_coefficients[24] = 0.01;

	return test_coefficients;
}


int main(int argc, char **argv)
{
	// should be preloaded from a video config file
	int width = 667;
	int height = 1000;
	int memsize_uchar3 = width * height * sizeof(uchar3);
	int memsize_uchar4 = width * height * sizeof(uchar4);

	// Gaussian blur coefficients and calculation
	int blur_radius = 2;
	int num_coeff = (blur_radius + 1) * (blur_radius + 1);
	float *h_blur_coeff = calculate_blur_coefficients(blur_radius);

	float *d_blur_coeff;
	cudaMalloc((void**)&d_blur_coeff, num_coeff * sizeof(float));
	cudaMemcpy(d_blur_coeff, h_blur_coeff, num_coeff * sizeof(float), cudaMemcpyHostToDevice);

	cudaDeviceProp  prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&dev, &prop);

	// tell CUDA which dev we will be using for graphic interop
	// from the programming guide:  Interoperability with OpenGL
	//     requires that the CUDA device be specified by
	//     cudaGLSetGLDevice() before any other runtime calls.


	cudaGLSetGLDevice(dev);

	// these GLUT calls need to be made before the other OpenGL
	// calls, else we get a seg fault
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("bitmap");
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	//not in tutorial, otherwise crashes
	if (GLEW_OK != glewInit()) { return 1; }
	while (GL_NO_ERROR != glGetError());

	// the first three are standard OpenGL, the 4th is the CUDA reg 
	// of the bitmap these calls exist starting in OpenGL 1.5
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, memsize_uchar4, NULL, GL_DYNAMIC_DRAW_ARB);

	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);

	uchar4* d_render_final;
	cudaMalloc((void**)&d_render_final, memsize_uchar4);

	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
	size_t  size;

	dim3 blockSize(32, 32);
	int bx = (width + 32 - 1) / 32;
	int by = (height + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);

	int morphing_param = 0;

	short *d_raster1;
	cudaMalloc((void**)&d_raster1, width * height * sizeof(short));

	short *d_raster2;
	cudaMalloc((void**)&d_raster2, width * height * sizeof(short));
	uchar3 * d_in_1;
	cudaMalloc((void**)&d_in_1, memsize_uchar3);
	uchar3 * d_in_2;
	cudaMalloc((void**)&d_in_2, memsize_uchar3);
	uchar3 * d_out_1;
	cudaMalloc((void**)&d_out_1, memsize_uchar3);

	uchar3 * d_out_2;
	cudaMalloc((void**)&d_out_2, memsize_uchar3);

	int * h_error_tracker = new int[3];
	h_error_tracker[0] = 0;
	h_error_tracker[1] = 0;
	h_error_tracker[2] = 0;
	int * d_error_tracker;
	cudaMalloc((void**)&d_error_tracker, 3 * sizeof(int));
	cudaMemcpy(d_error_tracker, h_error_tracker, 3 * sizeof(int), cudaMemcpyHostToDevice);

	for (;;) {


		auto t1 = std::chrono::high_resolution_clock::now();

		string img_path_1 = "../../data_store/binary/david_1.bin";
		string img_path_2 = "../../data_store/binary/david_2.bin";
		string raster1_path = "../../data_store/raster/rasterA_david.bin";
		string raster2_path = "../../data_store/raster/rasterB_david.bin";
		string affine_path = "../../data_store/affine/affine_david.bin";

		// BINARY IMAGE READ
		int length_1 = 0;
		int length_2 = 0;
		int width_1 = 0;
		int width_2 = 0;
		int height_1 = 0;
		int height_2 = 0;
		uchar3 *h_in_1 = read_uchar3_array(img_path_1, length_1, width_1, height_1);
		uchar3 *h_in_2 = read_uchar3_array(img_path_2, length_2, width_2, height_2);

		// RASTER READ
		int num_pixels_1 = 0;
		int num_pixels_2 = 0;
		short *h_raster1 = read_short_array(raster1_path, num_pixels_1);
		short *h_raster2 = read_short_array(raster2_path, num_pixels_2);

		// AFFINE READ
		int num_floats = 0;
		float *h_affine_data = read_float_array(affine_path, num_floats);
		int num_triangles = num_floats / 12;

		if (height_1 != height_2 || width_1 != width_2) {
			cout << "Incompatible image sizes. Program will now crash.\n";
			exit(-1);
		}

		//--Sending the data to the GPU memory
		cout << "declaring device data-structures..." << endl;

		float * d_affine_data;
		cudaMalloc((void**)&d_affine_data, num_floats * sizeof(float));
		cudaMemcpy(d_affine_data, h_affine_data, num_floats * sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(d_raster1, h_raster1, width * height * sizeof(short), cudaMemcpyHostToDevice);
		cudaMemcpy(d_raster2, h_raster2, width * height * sizeof(short), cudaMemcpyHostToDevice);
		cudaMemcpy(d_in_1, h_in_1, memsize_uchar3, cudaMemcpyHostToDevice);
		cudaMemcpy(d_in_2, h_in_2, memsize_uchar3, cudaMemcpyHostToDevice);

		float tau = (float)(morphing_param % 200) * 0.005f;

		float reverse_tau = 1.0f - tau;
		int reversal_offset = 0;

		kernel2D_subpix << <gridSize, blockSize >> >(d_out_1, d_in_1, d_raster1, d_error_tracker, width, height, d_affine_data, 4, tau, false);
		kernel2D_subpix << <gridSize, blockSize >> >(d_out_2, d_in_2, d_raster2, d_error_tracker, width, height, d_affine_data, 4, reverse_tau, true);
		kernel2D_add << <gridSize, blockSize >> > (d_render_final, d_out_1, d_out_2, width, height, tau);
		flip_y << < gridSize, blockSize >> >(d_render_final, width, height);
		gaussian_blur << < gridSize, blockSize >> > (d_render_final, width, height, d_blur_coeff, blur_radius);

		reset_image << <gridSize, blockSize >> > (d_out_1, width, height);
		reset_image << <gridSize, blockSize >> > (d_out_2, width, height);

		cudaFree(d_affine_data);


		cudaGraphicsMapResources(1, &resource, NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&d_render_final, &size, resource);
		cudaGraphicsUnmapResources(1, &resource, NULL);

		morphing_param++;

		//Does not seem "necessary"
		cudaDeviceSynchronize();

		//only gluMainLoopEvent() seems necessary
		glutPostRedisplay();
		glutMainLoopEvent();

		free(h_in_1);
		free(h_in_2);
		free(h_raster1);
		free(h_raster2);
		free(h_affine_data);
		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "Total: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << endl;


		cudaMemcpy(h_error_tracker, d_error_tracker, 3 * sizeof(int), cudaMemcpyDeviceToHost);
		cout << "Raster index OOB: " << h_error_tracker[0] << endl;
		cout << "Pre-processing pixel OOB: " << h_error_tracker[1] << endl;
		cout << "Post-processing pixel OOB: " << h_error_tracker[2] << endl;
	}

	cudaFree(d_in_1);
	cudaFree(d_out_1);
	cudaFree(d_raster1);
	cudaFree(d_in_2);
	cudaFree(d_out_2);
	cudaFree(d_raster2);
	cudaFree(d_render_final);
	// set up GLUT and kick off main loop
	glutMainLoop();

}