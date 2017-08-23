# What is this?

This project takes aim at reproducing the results from the paper “Spatial and Temporal Interpolation of Multi-View Image Sequences”, by Tobias Gurdan, Martin R. Oswalk, Daniel Gurdan, and Daniel Cremers. 

Although we were not able to achieve their results, this projects tries to morph a video into another video of the same scene, the two videos being synchronized. To achieve this, mvv\_demo is run, where we extract keypoints from both videos frame by frame using AKAZE, filter them using first Lowe's ratio test and then RANSAC. 

The resulting keypoints are triangulated using a Delauney triangulation and affine transformations are computed.  This triangulation information is fed into the CUDA routine cuda\_demo, which then morphs the frames by applying the affine transformations onto the triangles, sending the corresponding textures forward from one video onto the other, finally displaying the frames directly from the GPU. 

mvv\_demo does the preprocessing of the video files and must be run first. This project has a lengthy runtime, as it takes roughly 1 hour to run per 1 minute of video (this was tested at 95fps, 1920x1080 resolution with the system specifications described above). When this project is run, it will process the video files frame-by-frame, showing its progress as console output to the user. This program populates the \data\_store\ folder within this repository.

cuda\_demo is the second project, which uses CUDA and GLUT to render the interpolated 4D video file in real-time. When this project is run, it will take the files generated in the \data\_store\ folder and show the output to the user. The camera position of the 4D video can be dynamically changed by the user with the 'a' and 'd' keys.

# System Requirements

This utility requires a Windows 10 environment with Visual Studio 2015 installed, as well as CUDA-capable NVIDIA graphics card. The program will run on a hard drive, but for optimal performance we recommend using a solid state drive (if you use a solid-state drive, install the repository to a directory on your SSD).

Additionally, due to the extreme size of uncompressed 4D video files, you will want to make sure you have at least 100GB of free space.

The program was tested and works well on the following specifications:
-Windows 10 (64-bit)
-Intel(R) Core i7-6700HQ CPU @ 2.60 GHz
-8.00GB RAM
-GeForce GTX 960M

# Installation Procedure

Step 1. Install Visual Studio 2015

Step 2. Install the Cuda SDK from https://developer.nvidia.com/cuda-downloads 

Step 3. Update your NVidia drivers.

Step 4. Install OpenCV 3.2.0.
This precise version of OpenCV must be installed, as version-specific .dll files are referenced in this project. Download the OpenCV .exe installer and extract directly to the "C:\" drive. This will create a folder "C:\opencv" with version 3.2.0 installed.

Step 5. Clone or download this repository to your local machine.

# Configuration Steps

At this point, you should be able to compile the two Visual Studio solutions mvv\_demo.sln and cuda\_demo.sln. These two solutions comprise the two components of the overall algorithm.

At this point, all you need to do before running the programs (mvv_demo first, then cuda\_demo to visualize the result) is to modify the settings.ini file located in root of this repository. The .ini file includes the following information, which must be filled out by the user:

Six of these parameters are straightforward:

data_store_path: absolute path of the data_store folder within the repository
video_path_1 = absolute path of the first video file
video_path_2 = absolute path of the first video file 
video_width = horizontal pixel resolution of the videos
video_height = vertical pixel resolution of the videos
framerate = FPS of the videos;

Please note that the videos that are being interpolated must have the same resolution and framerate. Importantly, they do not have to be the same duration. The syncing of the video is taken care of by the following two parameters:

start_offset = This gives an integer constant offset from the beginning of both videos. This can be set to zero if you want to start at the very beginning.
delay = The delay is a floating point number specifying the delay in seconds of the second video relative to the first video. In order to calculate the offset, separate the audio from the video files using ffmpeg, and then process the audio files with the MATLAB script audio_sync.m, provided in the repository.

If you shoot your own videos to process with this approach, we recommend an inwards camera angle between 20 and 30 degrees.

# Acknowledgements

This project would not have been possible without the use of numerous open-source libraries that were included as part of this project. A special thanks to the creators of:

AKAZE (Accelerated-KAZE Features with CUDA acceleration)
Niklas Bergström
https://github.com/nbergst/akaze

INIH (Simple .INI file parser in C)
Ben Hoyt
https://github.com/benhoyt/inih
