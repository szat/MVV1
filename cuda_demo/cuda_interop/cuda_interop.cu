// The following line starts the program without a console window.
// Comment this out when you want to debug the application.
//#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

#include <windows.h>


// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include "binary_read.h"
#include "video_kernel.h"

using namespace std;

#define REFRESH_DELAY     2 //ms
#define RELEASE_MODE true

#define WIDTH 667
#define HEIGHT 1000

float camera_pos = 0.0;

//TRY TO CALL GLUTPOSTREDISPLAY FROM A FOOR LOOP
GLuint  bufferObj;

cudaGraphicsResource *resource;
__device__ int counter;
__device__ volatile int param = 50;

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

static void arrow_func(int key, int x, int y)
{
	if (key == 102) { //left arrow
		if(camera_pos + 0.005f < 1)  camera_pos = camera_pos + 0.005f;
	} 
	else if (key == 100) { //right arrow
		if (camera_pos - 0.005f > 0) camera_pos = camera_pos - 0.005f;
	}
	printf("camera_pos %f\n", camera_pos);
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

float * calculate_blur_coefficients(int blur_radius, float blur_param) {
	/*
	time for some normalization
	normalization constraints for this gaussian:
	*/
	int num_coeff = blur_radius * 2 + 1;
	float *coefficients = new float[num_coeff];
	// this value must be positive or it will mess up the blur
	blur_param = abs(blur_param);

	// f(x) = a*e^(-x^2/b)
	// setting central value
	coefficients[blur_radius] = 1.0f;
	for (int i = 1; i <= blur_radius; i++) {
		float exponent = -1.0f * (float)i*(float)i / blur_param;
		float coeff = exp(exponent);
		coefficients[blur_radius - i] = coeff;
		coefficients[blur_radius + i] = coeff;
	}
	float non_normalized_total = 0;
	for (int i = 0; i < num_coeff; i++) {
		non_normalized_total += coefficients[i];
	}
	for (int j = 0; j < num_coeff; j++) {
		coefficients[j] = coefficients[j] / non_normalized_total;
	}
	return coefficients;
}

int main(int argc, char **argv)
{
	cout << "Program startup" << endl;
	if (RELEASE_MODE) {
		cout << "NDim is in release mode" << endl;
	}
	else {
		cout << "NDim is in debug mode" << endl;
	}
	// should be preloaded from a video config file
	int memsize_uchar3 = WIDTH * HEIGHT * sizeof(uchar3);
	int memsize_uchar4 = WIDTH * HEIGHT * sizeof(uchar4);

	// Gaussian blur coefficients and calculation
	int blur_radius = 5;
	// smaller numbere means more blur
	float blur_param = 1.25f;
	int num_coeff = (2 * blur_radius + 1);
	float *h_blur_coeff = calculate_blur_coefficients(blur_radius, blur_param);

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
	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("bitmap");
	//glutFullScreen();
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
	glutSpecialFunc(arrow_func);
	glutDisplayFunc(draw_func);

	uchar4* d_render_final;
	cudaMalloc((void**)&d_render_final, memsize_uchar4);

	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
	size_t  size;

	dim3 blockSize(32, 32);
	int bx = (WIDTH + 32 - 1) / 32;
	int by = (HEIGHT + 32 - 1) / 32;
	dim3 gridSize = dim3(bx, by);

	int morphing_param = 0;

	short *d_raster1;
	cudaMalloc((void**)&d_raster1, WIDTH * HEIGHT * sizeof(short));

	short *d_raster2;
	cudaMalloc((void**)&d_raster2, WIDTH * HEIGHT * sizeof(short));
	uchar3 * d_in_1;
	cudaMalloc((void**)&d_in_1, memsize_uchar3);
	uchar3 * d_in_2;
	cudaMalloc((void**)&d_in_2, memsize_uchar3);
	uchar3 * d_out_1;
	cudaMalloc((void**)&d_out_1, memsize_uchar3);

	uchar3 * d_out_2;
	cudaMalloc((void**)&d_out_2, memsize_uchar3);

	for (;;) {
		auto t1 = std::chrono::high_resolution_clock::now();
		string img_path_1 = "../../data_store/binary/frame1.bin";
		string img_path_2 = "../../data_store/binary/frame2.bin";
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
		//cout << "declaring device data-structures..." << endl;

		float * d_affine_data;
		cudaMalloc((void**)&d_affine_data, num_floats * sizeof(float));
		cudaMemcpy(d_affine_data, h_affine_data, num_floats * sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(d_raster1, h_raster1, WIDTH * HEIGHT* sizeof(short), cudaMemcpyHostToDevice);
		cudaMemcpy(d_raster2, h_raster2, WIDTH * HEIGHT * sizeof(short), cudaMemcpyHostToDevice);
		cudaMemcpy(d_in_1, h_in_1, memsize_uchar3, cudaMemcpyHostToDevice);
		cudaMemcpy(d_in_2, h_in_2, memsize_uchar3, cudaMemcpyHostToDevice);
	
		interpolate_frame(gridSize, blockSize, d_out_1, d_out_2, d_in_1, d_in_2, d_render_final, d_raster1, d_raster2, WIDTH, HEIGHT, d_affine_data, 4, camera_pos);
		flip_image(gridSize, blockSize, d_render_final, WIDTH, HEIGHT);
		gaussian_2D_blur(gridSize, blockSize, d_render_final, WIDTH, HEIGHT, d_blur_coeff, blur_radius);
		reset_canvas(gridSize, blockSize, d_out_1, WIDTH, HEIGHT);
		reset_canvas(gridSize, blockSize, d_out_2, WIDTH, HEIGHT);

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
		//std::cout << "Total: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << endl;
		//std::cout << "Morphing param is " << morphing_param << endl;
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