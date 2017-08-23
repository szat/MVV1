# mvv_v1
Multi-view video v1

Steps for Visual Studio 2015 environment:

Install OpenCV:
Download link: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.2.0/opencv-3.2.0-vc14.exe/download
Install to C:\opencv directory

Note: The opencv installer specified in the above download link has precompiled lib/bin that can be used in 

Steps for creating a first OpenCV solution (.sln) and project (.csproj) for the first time:
1. Go to File --> New --> Project, and select Project.
2. New Project window appears, select Win32 Console Application (Visual C++).
3. Enter the name of the project into field "name", click "OK".
4. In the Win32 Application Wizard, click Next.
5. In the Win32 Application Wizard, deselect Precompiled header, deselect Security, Development Lifecycle (SDL) checks.
6. In the Win32 Application Wizard, click Finish.
7. In the Solution Explorer, select the newly created project, right-click and select Properties. 
8. In the Property Pages, go to C/C++ --> General, select the field "Additional Include Directories" and enter "C:\opencv\build\include;"
9. In the Property Pages, go to Linker --> Input, select the field "Additional Dependencies" and enter "opencv_world320.lib;"
10. In the Property Pages, go to Linker --> General, select the field "Additional Library Directories" and enter "C:\opencv\build\x64\vc14\lib;"
11. In the Property Pages, click "OK".
12. Finally navigate in a normal window browser to "C:\opencv\build\x64\vc14\bin;" and copy the files "opencv_world320.dll" and "opencv_world320d.dll", navigate to the folder containing the solution, and copy the two files into "...\x64\Debug".

======================================================================
******** IMPORTANT NOTIFICATION REGARDING THE NEXT STEP **************
======================================================================

We quickly realized that this is NOT the correct way to go about it. Each project should NOT be a separate .exe.
It is very difficult to incorporate different .exe's into one program. What we actually want (and what we have started doing)
is to have one main project created in the way described below, and the rest should be libraries referenced by the main project.

======================================================================

Steps for creating a new project (.csproj) in the solution, once you have a project that is already working:
1. Open the solution (.sln) in which the team is working.
2. In the Solution Explorer select the solution, right click and go to Add --> New Project...
3. Project creator window appears, select Win32 Console Application (Visual C++).
4. Enter the name of the project into field "name", click "OK".
5. In the Win32 Application Wizard, click "Next".
6. In the Win32 Application Wizard, deselect Precompiled header, deselect Security, Development Lifecycle (SDL) checks.
7. In the Win32 Application Wizard, click "Finish".
8. In the Solution Explorer, select the new project, right-click and select Properties.
9. The project Property Pages will pop up, which has the nice feature that if you select another project in the Solution Explorer, then the Property Pages is updated to the new project; however, as you navigate away from one project's Property Pages, Visual Studio will ask whether to save. 
10. In the Solution Explorer, select another project that is already working with OpenCV.
11. In the Property Pages, go to C/C++ --> General, select and copy the field "Additional Include Directories".
12. In the Solution Explorer, select the new project, go back to Property Pages, and paste into "Additional Include Directories". 
13. Int the Solution Explorer, select the other project, Property Pages will prompt you whether you want to save the changes you've made in the property pages, click Yes. 
14. Do steps 10 to 13 for the fields Linker --> General --> Additional Linker Directories, and Linker --> Input --> Additional Dependencies
15. In order to work in your new project, in the Solution Explorer, select the new project, right-click and select "Set as Startup Project".

Git procedure:

Start of every day (each person):
1. git pull

End of every day (each person):
1. git add .
2. git commit -m "Useful Comment"
3. git pull
4. resolve any merge conflicts that appear by editing the files manually
5. git push




2017-08-07: mvv_v1.06 stable build


Style conventions
-----------------

Variable, function, class names, commit comments and branch names should be descriptive, but as short as possible.

For .cpp code:

Variable names (camelCase):
    -exampleVariable
Function and class names (CapitalCase):
    -example_function
    -ExampleClass
Comments:
    -Keep the amount of comments to a minimum.
    -Comments should be short but descriptive.
    -Comments like "adding loop to iterate here" are worse than useless
    -Outdated comments are worse than useless. Comments should be updated with the code, or else deleted.
    
For all files:
    -File names should use all lowercase and underscores (eg. example_file.cpp
    
For source control:
    -Commits such as "feature added", or "bug fixed" are again, worse than useless
    -Keep commit names under 20 characters

To create a new library:
-Go to the solution explorer, right click on the solution and click "Add new project"
-Add new win32 console application, but select under application type "static library", and uncheck 
"precompiled header" and "SDL checks"

In the main project (core), right click properties, and create a new entry in additional include directories titled "..\your_library_name_here"

Additionally, add a reference to the new library under the main project (core).

Add all of the headers you need from the new library by selecting #include<library_file_1.h>
#include<library_file_2.h>
etc.

HEADERS:

When you create a new .cpp file, it is best to create it by going to your solution and selecting Add -> Class,
because it auto-creates a header file which is linked to the .cpp. Otherwise, you might have trouble getting
the header and .cpp to link correctly.

Additionally, although you do seem to need to put the include statements in your headers:

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

(maybe there is a way to do this in a centralized way so that it doesn't have to be in every header, but I'm 
not sure. At least now we only have the .dlls loaded in one project)

TODO: I know there is a way to only require these includes in one global header, but I'm not sure how that
works. There are some subtleties here that I'm not fully aware of -Danny

However, do NOT include namespaces in your header files. Every time you use a namespace, refer to it 
directly (for example, std::vector or cv::Rect)

You can use namespaces in your cpp files.

======================================================================
******** INSTALLATION PROCEDURE **************
======================================================================
Step 1. Install Visual Studio 2015
Step 2. Install the Cuda SDK from https://developer.nvidia.com/cuda-downloads 
Step 3. Update the Cuda drivers 
Step 4. Send Danny 50$
Step 5. Set startup project to be GPU wrapper