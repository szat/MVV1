////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
This example demonstrates how to use the Cuda OpenGL bindings to
dynamically modify a vertex buffer using a Cuda kernel.

The steps are:
1. Create an empty vertex buffer object (VBO)
2. Register the VBO with Cuda
3. Map the VBO for writing from Cuda
4. Run Cuda kernel to modify the vertex positions
5. Unmap the VBO
6. Render the results using OpenGL

Host code
*/

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

#define WIDTH 667 //size of david_1.jpg
#define HEIGHT 1000
#define REFRESH_DELAY     2 //ms

//TRY TO CALL GLUTPOSTREDISPLAY FROM A FOOR LOOP

GLuint  bufferObj;
cudaGraphicsResource *resource;
__device__ int counter;
__device__ volatile int param = 100;


__global__
void kernel2D_add(uchar4* d_output, uchar4* d_input_1, uchar4* d_input_2, int w, int h, float tau) {
	//tau is from a to b
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int raster_index = (row * w + col);

	// should not need to do this check if everything is good, must be an extra pixel
	if (raster_index >= w * h) return;
	if ((row >= h) || (col >= w)) return;


	if (d_input_1[raster_index].x == 0 && d_input_1[raster_index].y == 0 && d_input_1[raster_index].z == 0) {
		d_output[raster_index] = d_input_2[raster_index];
	}
	else if (d_input_2[raster_index].x == 0 && d_input_2[raster_index].y == 0 && d_input_2[raster_index].z == 0) {
		d_output[raster_index] = d_input_1[raster_index];
	}
	else {
		d_output[raster_index].x = tau*d_input_1[raster_index].x + (1 - tau)*d_input_2[raster_index].x;
		d_output[raster_index].y = tau*d_input_1[raster_index].y + (1 - tau)*d_input_2[raster_index].y;
		d_output[raster_index].z = tau*d_input_1[raster_index].z + (1 - tau)*d_input_2[raster_index].z;
	}
}


__global__ void kernel(uchar4 *ptr, uchar4* d_img_ptr, int w, int h, int param) {
	// map from threadIdx/BlockIdx to pixel position
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c;
	if ((r >= h) || (c >= w)) return;

	// accessing uchar4 vs unsigned char*
	ptr[i].x = (d_img_ptr[i].x + param) % 255;
	ptr[i].y = d_img_ptr[i].y;
	ptr[i].z = (d_img_ptr[i].z + param) % 200;
	ptr[i].w = d_img_ptr[i].w;
}

__global__ void kernel_2(uchar4 *ptr, int w, int h, int param) {
	// map from threadIdx/BlockIdx to pixel position
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c;
	if ((r >= h) || (c >= w)) return;

	//atomicAdd(&counter, 1);

	// accessing uchar4 vs unsigned char*
	ptr[i].x = ptr[i].x + param;
	ptr[i].y = ptr[i].y;
	ptr[i].z = ptr[i].z + param;
	ptr[i].w = ptr[i].w;
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

	glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
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

int main(int argc, char **argv)
{
	cudaDeviceProp  prop;
	int dev;

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&dev, &prop);
	cudaGLSetGLDevice(dev);

	// these GLUT calls need to be made before the other OpenGL
	// calls, else we get a seg fault
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("bitmap");
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	//not in tutorial, otherwise crashes
	if (GLEW_OK != glewInit()) { return 1; }
	while (GL_NO_ERROR != glGetError());

	// the first three are standard OpenGL, the 4th is the CUDA reg 
	// of the bitmap these calls exist starting in OpenGL 1.5
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIDTH * HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW_ARB);

	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);

	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
	
	/*
	int addXdir = 1;
	int * devAddXdir;
	cudaMalloc((void**)&devAddXdir, sizeof(int));
	cudaMemcpy(devAddXdir, &addXdir, sizeof(int), cudaMemcpyHostToDevice);
	*/

	uchar4* d_img_ptr_1;
	cudaMalloc((void**)&d_img_ptr_1, WIDTH*HEIGHT * sizeof(uchar4));

	uchar4* d_img_ptr_2;
	cudaMalloc((void**)&d_img_ptr_2, WIDTH*HEIGHT * sizeof(uchar4));

	uchar4* d_render;
	cudaMalloc((void**)&d_render, WIDTH*HEIGHT * sizeof(uchar4));

	short *d_raster1;
	cudaMalloc((void**)&d_raster1, WIDTH * HEIGHT * sizeof(short));

	short *d_raster2;
	cudaMalloc((void**)&d_raster2, WIDTH * HEIGHT * sizeof(short));

	int counter = 0;

	dim3 blockSize(32, 32);
	int bx = (WIDTH + 32 - 1) / 32;
	int by = (HEIGHT + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);

	for (;;) {
		printf("Counter: %d \n", counter);
		counter++;

		auto t1 = std::chrono::high_resolution_clock::now();


		// IMAGE READ
		string img_path_1 = "../../data_store/binary/david_1.bin";
		int length_1 = 0;
		int width_1 = 0;
		int height_1 = 0;
		uchar4* h_img_ptr_1 = read_uchar4_array(img_path_1, length_1, width_1, height_1);
		cudaMemcpy(d_img_ptr_1, h_img_ptr_1, WIDTH*HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);

		string img_path_2 = "../../data_store/binary/david_2.bin";
		int length_2 = 0;
		int width_2 = 0;
		int height_2 = 0;
		uchar4* h_img_ptr_2 = read_uchar4_array(img_path_2, length_2, width_2, height_2);
		cudaMemcpy(d_img_ptr_2, h_img_ptr_2, WIDTH*HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);

		
		// RASTER READ
		string raster1_path = "../../data_store/raster/rasterA.bin";
		int num_pixels_1 = 0;
		short *h_raster1 = read_short_array(raster1_path, num_pixels_1);
		cudaMemcpy(d_raster1, h_raster1, WIDTH * HEIGHT * sizeof(short), cudaMemcpyHostToDevice);

		string raster2_path = "../../data_store/raster/rasterB.bin";
		int num_pixels_2 = 0;
		short *h_raster2 = read_short_array(raster2_path, num_pixels_2);
		cudaMemcpy(d_raster2, h_raster2, WIDTH * HEIGHT * sizeof(short), cudaMemcpyHostToDevice);
		

		/*
		// AFFINE READ
		string affine_path = "../../data_store/affine/affine_1.bin";
		int num_floats = 0;
		float *h_affine_data = read_float_array(affine_path, num_floats);
		int num_triangles = num_floats / 12;
		float * d_affine_data;
		cudaMalloc((void**)&d_affine_data, num_floats * sizeof(float));
		cudaMemcpy(d_affine_data, h_affine_data, num_floats * sizeof(float), cudaMemcpyHostToDevice);
		free(h_affine_data);
		cudaFree(d_affine_data);
		*/

		cudaGraphicsMapResources(1, &resource, NULL);
		size_t  size;
		cudaGraphicsResourceGetMappedPointer((void**)&d_render, &size, resource);

		kernel_2 << <gridSize, blockSize >> >(d_img_ptr_1, WIDTH, HEIGHT, counter);
		kernel_2 << <gridSize, blockSize >> >(d_img_ptr_2, WIDTH, HEIGHT, counter);
		kernel2D_add << <gridSize, blockSize >> > (d_render, d_img_ptr_1, d_img_ptr_2, WIDTH, HEIGHT, 0.5);

		cudaGraphicsUnmapResources(1, &resource, NULL);
		
		free(h_img_ptr_1);
		free(h_img_ptr_2);
		

		//Does not seem "necessary"
		cudaDeviceSynchronize();

		auto t2 = std::chrono::high_resolution_clock::now();
		std::cout << "took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()	<< " ms\n";

		//only gluMainLoopEvent() seems necessary
		glutPostRedisplay();
		glutMainLoopEvent();
	}

	// set up GLUT and kick off main loop
	glutMainLoop();

}