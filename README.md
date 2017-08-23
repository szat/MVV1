# What is this?

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

mvv\_demo does the preprocessing of the video files and must be run first. This project has a lengthy runtime, as it takes roughly 1 hour to run per 1 minute of video (this was tested at 95fps, 1920x1080 resolution with the system specifications described above). When this project is run, it will process the video files frame-by-frame, showing its progress as console output to the user. This program populates the \data\_store\ folder within this repository.

cuda\_demo is the second project, which uses CUDA and GLUT to render the interpolated 4D video file in real-time. When this project is run, it will take the files generated in the \data\_store\ folder and show the output to the user. The camera position of the 4D video can be dynamically changed by the user with the 'a' and 'd' keys.
