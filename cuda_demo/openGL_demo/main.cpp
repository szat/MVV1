#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <iostream>
#include <string>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "interactions.h"

using namespace std;
using namespace cv;

// texture and pixel objects
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

void render() {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
	kernelLauncher(d_out, W, H, loc);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

void display() {
	//render();
	drawTexture();
	glutSwapBuffers();
	glutPostRedisplay();
}

void initGLUT(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(W, H);
	glutCreateWindow(TITLE_STRING);
	glewInit();
}

 void initPixelBuffer() {
	 glGenBuffers(1, &pbo);
	 glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	 glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H * sizeof(GLubyte), 0, GL_STREAM_DRAW);
	 glGenTextures(1, &tex);
	 glBindTexture(GL_TEXTURE_2D, tex);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	 cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
 }

 void exitfunc() {
	 if (pbo) {
		 cudaGraphicsUnregisterResource(cuda_pbo_resource);
		 glDeleteBuffers(1, &pbo);
		 glDeleteTextures(1, &tex);
	 }
 }

 int main(int argc, char** argv) {
	 cout << "welcome to cuda_demo testing unit!" << endl;
	 cout << "loading 2 images with openCV, processing and adding them with cuda (grayscale)." << endl;

	 string img1_path = "../../data_store/images/david_1.jpg";
	 string img2_path = "../../data_store/images/david_2.jpg";

	 // Initializing CUDA
	 Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
	 Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);

	 int tex_width = img1.size().width;
	 int tex_height = img1.size().height;

	 GLuint texId;
	 cudaGraphicsResource_t texRes;
	 // OpenGL buffer creation...
	 glGenTextures(1, &texId);
	 glBindTexture(GL_TEXTURE_2D, texId);
	 glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, tex_width, tex_height, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, 0);
	 glBindTexture(GL_TEXTURE_2D, 0);
	 
	 // Registration with CUDA.
	 cudaGraphicsGLRegisterImage(&texRes, texId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

	 cudaArray* texArray;
	 bool done = false;

	 while (!done)
	 {
		 cudaGraphicsMapResources(1, &texRes);
		 cudaGraphicsSubResourceGetMappedArray(&texArray, texRes, 0, 0);
		 //runCUDA(texArray);
		 cudaGraphicsUnmapResources(1, &texRes);
		 //runGL(texId);
	 }

	 /* //previous trial
	 printInstructions();
	 initGLUT(&argc, argv);
	 gluOrtho2D(0, W, H, 0);
	 glutKeyboardFunc(keyboard);
	 //glutSpecialFunc(handleSpecialKeypress);
	 //glutPassiveMotionFunc(mouseMove);
	 //glutMotionFunc(mouseDrag);

	 
	 //render();
	 uchar4 *d_out = 0;
	 cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	 cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
	 kernelLauncher(d_out, W, H, loc);
	 cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
	 

	 glutDisplayFunc(display);


	 //initPixelBuffer();
	 glGenBuffers(1, &pbo);
	 glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	 glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H * sizeof(GLubyte), 0, GL_STREAM_DRAW);
	 glGenTextures(1, &tex);
	 glBindTexture(GL_TEXTURE_2D, tex);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	 cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);


	 glutMainLoop();
	 atexit(exitfunc);
	 return 0;
	 */
	 return 0;
 }