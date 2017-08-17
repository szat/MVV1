
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void akaze_script(float akaze_thresh, const Mat& img_in, vector<KeyPoint>& kpts_out, Mat& desc_out) {
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	time_t tstart, tend;
	tstart = time(0);
	akaze->detectAndCompute(img_in, noArray(), kpts_out, desc_out);
	tend = time(0);
	cout << "akaze_wrapper(thr=" << akaze_thresh << ",[h=" << img_in.size().height << ",w=" << img_in.size().width << "]) finished in " << difftime(tend, tstart) << "s and found " << kpts_out.size() << " features." << endl;
}

