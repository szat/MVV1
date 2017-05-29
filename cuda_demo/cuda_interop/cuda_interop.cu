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

#define REFRESH_DELAY     2 //ms

//TRY TO CALL GLUTPOSTREDISPLAY FROM A FOOR LOOP

GLuint  bufferObj;
cudaGraphicsResource *resource;
__device__ int counter;
__device__ volatile int param = 50;


__global__ void shift_image(uchar4 *ptr, uchar4* d_img_ptr, int w, int h, int param) {
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

__global__ void morph_image(uchar4 *ptr, int w, int h, int param) {
	// map from threadIdx/BlockIdx to pixel position
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c;
	if ((r >= h) || (c >= w)) return;

	//atomicAdd(&counter, 1);

	// accessing uchar4 vs unsigned char*
	ptr[i].x = (ptr[i].x + 10 * param) % 255;
	ptr[i].y = ptr[i].y;
	ptr[i].z = (ptr[i].z + 10 * param) % 255;
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

int main(int argc, char **argv)
{
	// should be preloaded from a video config file
	int width = 667;
	int height = 1000;
	int memsize = width * height * sizeof(uchar4);

	// use calloc
	//uchar4* h_img_ptr = (uchar4*)calloc(0, sizeof(uchar4));
	
	uchar4* h_img_ptr = new uchar4[width * height];
	
	uchar4* d_img_ptr;
	cudaMalloc((void**)&d_img_ptr, memsize);
	cudaMemcpy(d_img_ptr, h_img_ptr, memsize, cudaMemcpyHostToDevice);


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
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, memsize, NULL, GL_DYNAMIC_DRAW_ARB);

	glutKeyboardFunc(key_func);
	//which one to pick? goes along with postrediplay
	//glutIdleFunc(draw_func);
	glutDisplayFunc(draw_func);

	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);

	cudaGraphicsMapResources(1, &resource, NULL);

	uchar4* d_img_display;
	size_t  size;

	cudaGraphicsResourceGetMappedPointer((void**)&d_img_display, &size, resource);

	dim3 blockSize(32, 32);
	int bx = (width + 32 - 1) / 32;
	int by = (height + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);
	shift_image << <gridSize, blockSize >> >(d_img_display, d_img_ptr, width, height, param);

	cudaGraphicsUnmapResources(1, &resource, NULL);

	/*
	int addXdir = 1;
	int * devAddXdir;
	cudaMalloc((void**)&devAddXdir, sizeof(int));
	cudaMemcpy(devAddXdir, &addXdir, sizeof(int), cudaMemcpyHostToDevice);
	*/
	int morphing_param = 0;
	for (;;) {
		

		string img_path_1 = "../../data_store/binary/david_1.bin";
		string img_path_2 = "../../data_store/binary/david_2.bin";

		int length_1 = 0;
		int width_1 = 0;
		int height_1 = 0;
		uchar4* h_onload_ptr = read_uchar4_array(img_path_1, length_1, width_1, height_1);
		cudaMemcpy(d_img_display, h_onload_ptr, memsize, cudaMemcpyHostToDevice);

		dim3 blockSize(32, 32);
		int bx = (width + 32 - 1) / 32;
		int by = (height + 32 - 1) / 32;
		dim3 gridSize = dim3(bx, by);

		cudaGraphicsMapResources(1, &resource, NULL);

		uchar4* d_img_display;
		size_t  size;

		cudaGraphicsResourceGetMappedPointer((void**)&d_img_display, &size, resource);


		morph_image << <gridSize, blockSize >> >(d_img_display, width, height, morphing_param);

		cudaGraphicsUnmapResources(1, &resource, NULL);

		free(h_onload_ptr);
		morphing_param++;
		//Does not seem "necessary"
		cudaDeviceSynchronize();

		//only gluMainLoopEvent() seems necessary
		glutPostRedisplay();
		glutMainLoopEvent();

	}

	// set up GLUT and kick off main loop
	glutMainLoop();

}