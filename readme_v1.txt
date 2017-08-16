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
