#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include <windows.h>

// OpenGL Graphics includes
#include <helper_gl.h>
#include <freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>


#include <iostream>
#include <cuda.h>
#include <conio.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "binary_io.h"

using namespace std;
using namespace cv;

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 667;
const unsigned int window_height = 1000;

const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	pos[y*width + x] = make_float4(u, w, v, 1.0f);
}


void launch_kernel(float4 *pos, unsigned int mesh_width,
	unsigned int mesh_height, float time)
{
	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(pos, mesh_width, mesh_height, time);
}

bool checkHW(char *name, const char *gpuType, int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy(name, deviceProp.name);

	if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findGraphicsGPU(char *name)
{
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[256], temp[256];

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
		printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

		if (bGraphics)
		{
			if (!bFoundGraphics)
			{
				strcpy(firstGraphicsName, temp);
			}

			nGraphicsGPU++;
		}
	}

	if (nGraphicsGPU)
	{
		strcpy(name, firstGraphicsName);
	}
	else
	{
		strcpy(name, "this hardware");
	}

	return nGraphicsGPU;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	char *ref_file = NULL;

	pArgc = &argc;
	pArgv = argv;

#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s starting...\n", sSDKsample);

	if (argc > 1)
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "file"))
		{
			// In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
			getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
		}
	}

	printf("\n");

	runTest(argc, argv, ref_file);

	printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();

	return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// command line mode only
	if (ref_file != NULL)
	{
		// This will pick the best possible CUDA capable device
		int devID = findCudaDevice(argc, (const char **)argv);

		// create VBO
		checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, mesh_width*mesh_height * 4 * sizeof(float)));

		// run the cuda part
		runAutoTest(devID, argv, ref_file);

		// check result of Cuda step
		checkResultCuda(argc, argv, vbo);

		cudaFree(d_vbo_buffer);
		d_vbo_buffer = NULL;
	}
	else
	{
		// First initialize OpenGL context, so we can properly set the GL for CUDA.
		// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
		if (false == initGL(&argc, argv))
		{
			return false;
		}

		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		if (checkCmdLineFlag(argc, (const char **)argv, "device"))
		{
			if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
			{
				return false;
			}
		}
		else
		{
			cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
		}

		// register callbacks
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
		atexit(cleanup);
#else
		glutCloseFunc(cleanup);
#endif

		// create VBO
		createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

		// run the cuda part
		runCuda(&cuda_vbo_resource);

		// start rendering mainloop
		glutMainLoop();
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	// execute the kernel
	//    dim3 block(8, 8, 1);
	//    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	//    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

	launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
	printf("sdkDumpBin: <%s>\n", filename);
	FILE *fp;
	FOPEN(fp, filename, "wb");
	fwrite(data, bytes, 1, fp);
	fflush(fp);
	fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
	char *reference_file = NULL;
	void *imageData = malloc(mesh_width*mesh_height*sizeof(float));

	// execute the kernel
	launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

	cudaDeviceSynchronize();
	getLastCudaError("launch_kernel failed");

	checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost));

	sdkDumpBin2(imageData, mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
	reference_file = sdkFindFilePath(ref_file, argv[0]);

	if (reference_file &&
		!sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
			mesh_width*mesh_height*sizeof(float),
			MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
	{
		g_TotalErrors++;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();

	g_fAnim += 0.01f;

	sdkStopTimer(&timer);
	computeFPS();
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27) :
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
	if (!d_vbo_buffer)
	{
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

		// map buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float *data = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

		// check result
		if (checkCmdLineFlag(argc, (const char **)argv, "regression"))
		{
			// write file for regression test
			sdkWriteFile<float>("./data/regression.dat",
				data, mesh_width * mesh_height * 3, 0.0, false);
		}

		// unmap GL buffer object
		if (!glUnmapBuffer(GL_ARRAY_BUFFER))
		{
			fprintf(stderr, "Unmap buffer failed.\n");
			fflush(stderr);
		}

		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
			cudaGraphicsMapFlagsWriteDiscard));

		SDK_CHECK_ERROR_GL();
	}
}

/*
typedef std::chrono::high_resolution_clock Clock;

__global__
void kernel2D(uchar* d_output, uchar* d_input, int w, int h, float * d_affineData)
{
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	int i = r * w + c;

	if ((r >= h) || (c >= w)) return;

	int new_c = (int)(d_affineData[0] * (float)c + d_affineData[1] * (float)r + d_affineData[2]);
	int new_r = (int)(d_affineData[3] * (float)c + d_affineData[4] * (float)r + d_affineData[5]);

	if ((new_r >= h) || (new_c >= w) || (new_r < 0) || (new_c < 0)) return;

	int new_i = new_r * w + new_c;
	d_output[new_i] = d_input[i];
}

__global__
void kernel2D_subpix(uchar* d_output, uchar* d_input, short* d_raster1, int w, int h, float * d_affineData, int subDiv, float tau, bool reverse)
{
	if (tau > 1 || tau < 0) return;

	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int raster_index = (row * w + col);
	int color_index = raster_index * 3;

	// should not need to do this check if everything is good, must be an extra pixel
	if (color_index + 2 >= w * h * 3) return;
	if ((row >= h) || (col >= w)) return;
	
	short affine_index = d_raster1[raster_index];
	short offset = affine_index * 12;
	if (reverse) {
		offset += 6;
	}
	if (affine_index != 0) {
		float diff = 1 / (float)subDiv;
		for (int i = 0; i < subDiv; i++) {
			for (int j = 0; j < subDiv; j++) {
				int new_c = (int)(((1-tau) + tau*d_affineData[offset]) * (float)(col - 0.5 + (diff * i)) + (tau * d_affineData[offset + 1]) * (float)(row - 0.5 + (diff * j)) + (tau * d_affineData[offset + 2]));
				int new_r = (int)((tau * d_affineData[offset + 3]) * (float)(col - 0.5 + (diff * i)) + ((1-tau) + tau * d_affineData[offset + 4]) * (float)(row - 0.5 + (diff * j)) + (tau * d_affineData[offset + 5]));
				if ((new_r >= h) || (new_c >= w) || (new_r < 0) || (new_c < 0)) return;
				int new_i = new_r * w + new_c;
				int new_index = new_i * 3;
				d_output[new_index] = d_input[color_index];
				d_output[new_index + 1] = d_input[color_index+1];
				d_output[new_index + 2] = d_input[color_index+2];
			}
		}
	}
	
	
}

__global__
void kernel2D_add(uchar* d_output, uchar* d_input_1, uchar* d_input_2, int w, int h, float tau) {
	//tau is from a to b
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int raster_index = (row * w + col);
	int color_index = raster_index * 3;

	// should not need to do this check if everything is good, must be an extra pixel
	if (color_index + 2 >= w * h * 3) return;
	if ((row >= h) || (col >= w)) return;

	
	for (int i = 0; i < 3; i++) {
		if (d_input_1[color_index + i] == 0) {
			d_output[color_index + i] = d_input_2[color_index + i];
		}
		else if (d_input_2[color_index + i] == 0) {
			d_output[color_index + i] = d_input_1[color_index + i];
		}
		else {
			d_output[color_index + i] = tau*d_input_1[color_index + i] + (1 - tau)*d_input_2[color_index + i];
		}
	}
	

}

Mat trial_binary_render(uchar *image, int width, int height) {
	Mat img(height, width, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			int index = (i * width + j) * 3;

			uchar r = image[index];
			uchar g = image[index + 1];
			uchar b = image[index + 2];

			Vec3b color = Vec3b(r, g, b);

			img.at<Vec3b>(i, j) = color;
		}
	}

	return img;
}

int main(int argc, char ** argv) {
	cout << "welcome to cuda_demo testing unit!" << endl;
	cout << "loading 2 images with openCV, processing and adding them with cuda (grayscale)." << endl;

	// Initializing CUDA
	uchar *h_tester = new uchar[1];
	h_tester[0] = (uchar)0;
	uchar *d_tester;
	cudaMalloc((void**)&d_tester, sizeof(uchar));
	cudaMemcpy(d_tester, h_tester, sizeof(uchar), cudaMemcpyHostToDevice);
	cudaFree(d_tester);

	auto t1 = std::chrono::high_resolution_clock::now();

	string img_path_1 = "../../data_store/binary/david_1.bin";
	string img_path_2 = "../../data_store/binary/david_2.bin";
	string raster1_path = "../../data_store/raster/rasterA.bin";
	string raster2_path = "../../data_store/raster/rasterB.bin";
	string affine_path = "../../data_store/affine/affine_1.bin";

	// BINARY IMAGE READ
	int length_1 = 0;
	int length_2 = 0;
	int width_1 = 0;
	int width_2 = 0;
	int height_1 = 0;
	int height_2 = 0;
	uchar *h_in_1 = read_uchar_array(img_path_1, length_1, width_1, height_1);
	uchar *h_in_2 = read_uchar_array(img_path_2, length_2, width_2, height_2);
	
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

	int W = width_1;
	int H = height_1;
	int mem_alloc = W * H * 3 * sizeof(uchar);

	uchar *h_out_1;
	uchar *h_out_2;
	uchar *h_sum;
	
	// there must be a faster way to initialize these arrays to all zeros
	
	h_out_1 = (uchar*)malloc(mem_alloc);
	for (int j = 0; j < W*H*3; j++) h_out_1[j] = 0;

	h_out_2 = (uchar*)malloc(mem_alloc);
	for (int j = 0; j < W*H*3; j++) h_out_2[j] = 0;

	h_sum = (uchar*)malloc(mem_alloc);
	for (int j = 0; j < W*H*3; j++) h_sum[j] = 0;
	
	//--Sending the data to the GPU memory
	cout << "declaring device data-structures..." << endl;

	float * d_affine_data;
	cudaMalloc((void**)&d_affine_data, num_floats * sizeof(float));
	cudaMemcpy(d_affine_data, h_affine_data, num_floats * sizeof(float), cudaMemcpyHostToDevice);

	short *d_raster1;
	cudaMalloc((void**)&d_raster1, W * H * sizeof(short));
	cudaMemcpy(d_raster1, h_raster1, W * H * sizeof(short), cudaMemcpyHostToDevice);

	short *d_raster2;
	cudaMalloc((void**)&d_raster2, W * H * sizeof(short));
	cudaMemcpy(d_raster2, h_raster2, W * H * sizeof(short), cudaMemcpyHostToDevice);

	uchar * d_in_1;
	cudaMalloc((void**)&d_in_1, mem_alloc);
	cudaMemcpy(d_in_1, h_in_1, mem_alloc, cudaMemcpyHostToDevice);

	uchar * d_out_1;
	cudaMalloc((void**)&d_out_1, mem_alloc);
	cudaMemcpy(d_out_1, h_out_1, mem_alloc, cudaMemcpyHostToDevice);

	uchar * d_in_2;
	cudaMalloc((void**)&d_in_2, mem_alloc);
	cudaMemcpy(d_in_2, h_in_2, mem_alloc, cudaMemcpyHostToDevice);

	uchar * d_out_2;
	cudaMalloc((void**)&d_out_2, mem_alloc);
	cudaMemcpy(d_out_2, h_out_2, mem_alloc, cudaMemcpyHostToDevice);

	uchar * d_sum;
	cudaMalloc((void**)&d_sum, mem_alloc);
	cudaMemcpy(d_sum, h_sum, mem_alloc, cudaMemcpyHostToDevice);

	dim3 blockSize(32, 32);
	int bx = (W + 32 - 1) / 32;
	int by = (H + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);

	float tau = 0.5f;
	float reverse_tau = 1.0f - tau;
	int reversal_offset = 0;


	kernel2D_subpix << <gridSize, blockSize >> >(d_out_1, d_in_1, d_raster1, W, H, d_affine_data, 4, tau, false);
	kernel2D_subpix << <gridSize, blockSize >> >(d_out_2, d_in_2, d_raster2, W, H, d_affine_data, 4, reverse_tau, true);
	kernel2D_add << <gridSize, blockSize >> > (d_sum, d_out_1, d_out_2, W, H, tau);


	cudaMemcpy(h_out_1, d_out_1, mem_alloc, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out_2, d_out_2, mem_alloc, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sum, d_sum, mem_alloc, cudaMemcpyDeviceToHost);

	cudaFree(d_in_1);
	cudaFree(d_out_1);
	cudaFree(d_raster1);
	cudaFree(d_in_2);
	cudaFree(d_out_2);
	cudaFree(d_raster2);
	cudaFree(d_affine_data);
	cudaFree(d_sum);


	auto t2 = std::chrono::high_resolution_clock::now();
	std::cout << "write short took "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
		<< " milliseconds\n";

	//trial_binary_render(h_sum, W, H);
	Mat img1_initial = trial_binary_render(h_in_1, W, H);
	Mat img2_initial = trial_binary_render(h_in_2, W, H);
	Mat img1_final = trial_binary_render(h_out_1, W, H);
	Mat img2_final = trial_binary_render(h_out_2, W, H);
	Mat img_final = trial_binary_render(h_sum, W, H);



	return 0;
}
*/