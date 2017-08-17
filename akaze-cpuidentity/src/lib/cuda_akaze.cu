#include <opencv2/features2d/features2d.hpp>
#include "cuda_akaze.h"
#include "cudautils.h"

#define CONVROW_W 160
#define CONVCOL_W 32
#define CONVCOL_H 40
#define CONVCOL_S 8

#define SCHARR_W 32
#define SCHARR_H 16

#define NLDSTEP_W 32
#define NLDSTEP_H 13

#define ORIENT_S (13 * 16)
#define EXTRACT_S 64

#define MAX_BLOCK 512


__device__ __constant__ float d_Kernel[21];
__device__ unsigned int d_PointCounter[1];
__device__ unsigned int d_ExtremaIdx[16];

__device__ __constant__ int comp_idx_1[61 * 8];
__device__ __constant__ int comp_idx_2[61 * 8];

cudaStream_t copyStream;

//__device__ __constant__ float norm_factors[29];

#if 1
#define CHK
#else
#define CHK cudaDeviceSynchronize(); \
    { \
    cudaError_t cuerr = cudaGetLastError(); \
    if (cuerr) {							\
	std::cout << "Cuda error " << cudaGetErrorString(cuerr) << ". at " << __FILE__ << ":" << __LINE__ << std::endl; \
    } \
    }
#endif

void WaitCuda() {
    cudaStreamSynchronize(copyStream);
}

struct Conv_t {
  float *d_Result;
  float *d_Data;
  int width;
  int pitch;
  int height;
};

template <int RADIUS>
__global__ void ConvRowGPU(struct Conv_t s) {
  //__global__ void ConvRowGPU(float *d_Result, float *d_Data, int width, int
  //pitch, int height) {
  __shared__ float data[CONVROW_W + 2 * RADIUS];
  const int tx = threadIdx.x;
  const int minx = blockIdx.x * CONVROW_W;
  const int maxx = min(minx + CONVROW_W, s.width);
  const int yptr = blockIdx.y * s.pitch;
  const int loadPos = minx + tx - RADIUS;
  const int writePos = minx + tx;

  if (loadPos < 0)
    data[tx] = s.d_Data[yptr];
  else if (loadPos >= s.width)
    data[tx] = s.d_Data[yptr + s.width - 1];
  else
    data[tx] = s.d_Data[yptr + loadPos];
  __syncthreads();
  if (writePos < maxx && tx < CONVROW_W) {
    float sum = 0.0f;
    for (int i = 0; i <= (2 * RADIUS); i++) sum += data[tx + i] * d_Kernel[i];
    s.d_Result[yptr + writePos] = sum;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Column convolution filter
///////////////////////////////////////////////////////////////////////////////
template <int RADIUS>
__global__ void ConvColGPU(struct Conv_t s) {
  //__global__ void ConvColGPU(float *d_Result, float *d_Data, int width, int
  //pitch, int height) {
  __shared__ float data[CONVCOL_W * (CONVCOL_H + 2 * RADIUS)];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int miny = blockIdx.y * CONVCOL_H;
  const int maxy = min(miny + CONVCOL_H, s.height) - 1;
  const int totStart = miny - RADIUS;
  const int totEnd = maxy + RADIUS;
  const int colStart = blockIdx.x * CONVCOL_W + tx;
  const int colEnd = colStart + (s.height - 1) * s.pitch;
  const int smemStep = CONVCOL_W * CONVCOL_S;
  const int gmemStep = s.pitch * CONVCOL_S;

  if (colStart < s.width) {
    int smemPos = ty * CONVCOL_W + tx;
    int gmemPos = colStart + (totStart + ty) * s.pitch;
    for (int y = totStart + ty; y <= totEnd; y += blockDim.y) {
      if (y < 0)
        data[smemPos] = s.d_Data[colStart];
      else if (y >= s.height)
        data[smemPos] = s.d_Data[colEnd];
      else
        data[smemPos] = s.d_Data[gmemPos];
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
  __syncthreads();
  if (colStart < s.width) {
    int smemPos = ty * CONVCOL_W + tx;
    int gmemPos = colStart + (miny + ty) * s.pitch;
    for (int y = miny + ty; y <= maxy; y += blockDim.y) {
      float sum = 0.0f;
      for (int i = 0; i <= 2 * RADIUS; i++)
        sum += data[smemPos + i * CONVCOL_W] * d_Kernel[i];
      s.d_Result[gmemPos] = sum;
      smemPos += smemStep;
      gmemPos += gmemStep;
    }
  }
}

template <int RADIUS>
double SeparableFilter(CudaImage &inimg, CudaImage &outimg, CudaImage &temp,
                       float *h_Kernel) {
  int width = inimg.width;
  int pitch = inimg.pitch;
  int height = inimg.height;
  float *d_DataA = inimg.d_data;

  float *d_DataB = outimg.d_data;
  float *d_Temp = temp.d_data;
  if (d_DataA == NULL || d_DataB == NULL || d_Temp == NULL) {
    printf("SeparableFilter: missing data\n");
    return 0.0;
  }
  // TimerGPU timer0(0);
  const unsigned int kernelSize = (2 * RADIUS + 1) * sizeof(float);
  safeCall(cudaMemcpyToSymbolAsync(d_Kernel, h_Kernel, kernelSize));

  dim3 blockGridRows(iDivUp(width, CONVROW_W), height);
  dim3 threadBlockRows(CONVROW_W + 2 * RADIUS);
  struct Conv_t s;
  s.d_Result = d_Temp;
  s.d_Data = d_DataA;
  s.width = width;
  s.pitch = pitch;
  s.height = height;
  ConvRowGPU<RADIUS> << <blockGridRows, threadBlockRows>>> (s);
  // checkMsg("ConvRowGPU() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  dim3 blockGridColumns(iDivUp(width, CONVCOL_W), iDivUp(height, CONVCOL_H));
  dim3 threadBlockColumns(CONVCOL_W, CONVCOL_S);
  s.d_Result = d_DataB;
  s.d_Data = d_Temp;
  ConvColGPU<RADIUS> << <blockGridColumns, threadBlockColumns>>> (s);
  // checkMsg("ConvColGPU() execution failed\n");
  // safeCall(cudaThreadSynchronize());

  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("SeparableFilter time =        %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

template <int RADIUS>
double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp,
               double var) {
  float kernel[2 * RADIUS + 1];
  float kernelSum = 0.0f;
  for (int j = -RADIUS; j <= RADIUS; j++) {
    kernel[j + RADIUS] = (float)expf(-(double)j * j / 2.0 / var);
    kernelSum += kernel[j + RADIUS];
  }
  for (int j = -RADIUS; j <= RADIUS; j++) kernel[j + RADIUS] /= kernelSum;
  return SeparableFilter<RADIUS>(inimg, outimg, temp, kernel);
}

double LowPass(CudaImage &inimg, CudaImage &outimg, CudaImage &temp, double var,
               int kernsize) {
  if (kernsize <= 5)
    return LowPass<2>(inimg, outimg, temp, var);
  else if (kernsize <= 7)
    return LowPass<3>(inimg, outimg, temp, var);
  else if (kernsize <= 9)
    return LowPass<4>(inimg, outimg, temp, var);
  else {
    if (kernsize > 11)
      std::cerr << "Kernels larger than 11 not implemented" << std::endl;
    return LowPass<5>(inimg, outimg, temp, var);
  }
}

__inline__ __device__
int fake_shfl_down(int val, int offset, int width = 32) {
	static __shared__ int shared[MAX_BLOCK];
	int lane = threadIdx.x % 32;

	shared[threadIdx.x] = val;
	__syncthreads();

	val = (lane + offset<width) ? shared[threadIdx.x + offset] : 0;
	__syncthreads();

	return val;
}

__global__ void Scharr(float *imgd, float *lxd, float *lyd, int width,
                       int pitch, int height) {
#define BW (SCHARR_W + 2)
  __shared__ float buffer[BW * (SCHARR_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * SCHARR_W + tx;
  int y = blockIdx.y * SCHARR_H + ty;
  int xp = (x == 0 ? 1 : (x > width ? width - 2 : x - 1));
  int yp = (y == 0 ? 1 : (y > height ? height - 2 : y - 1));
  buffer[ty * BW + tx] = imgd[yp * pitch + xp];
  __syncthreads();
  if (x < width && y < height && tx < SCHARR_W && ty < SCHARR_H) {
    float *b = buffer + (ty + 1) * BW + (tx + 1);
    float ul = b[-BW - 1];
    float ur = b[-BW + 1];
    float ll = b[+BW - 1];
    float lr = b[+BW + 1];
    lxd[y * pitch + x] = 3.0f * (lr - ll + ur - ul) + 10.0f * (b[+1] - b[-1]);
    lyd[y * pitch + x] = 3.0f * (lr + ll - ur - ul) + 10.0f * (b[BW] - b[-BW]);
  }
}

double Scharr(CudaImage &img, CudaImage &lx, CudaImage &ly) {
  // TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, SCHARR_W), iDivUp(img.height, SCHARR_H));
  dim3 threads(SCHARR_W + 2, SCHARR_H + 2);
  Scharr << <blocks, threads>>>
      (img.d_data, lx.d_data, ly.d_data, img.width, img.pitch, img.height);
  // checkMsg("Scharr() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("Scharr time          =        %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void Flow(float *imgd, float *flowd, int width, int pitch,
                     int height, DIFFUSIVITY_TYPE type, float invk) {
#define BW (SCHARR_W + 2)
  __shared__ float buffer[BW * (SCHARR_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * SCHARR_W + tx;
  int y = blockIdx.y * SCHARR_H + ty;
  int xp = (x == 0 ? 1 : (x > width ? width - 2 : x - 1));
  int yp = (y == 0 ? 1 : (y > height ? height - 2 : y - 1));
  buffer[ty * BW + tx] = imgd[yp * pitch + xp];
  __syncthreads();
  if (x < width && y < height && tx < SCHARR_W && ty < SCHARR_H) {
    float *b = buffer + (ty + 1) * BW + (tx + 1);
    float ul = b[-BW - 1];
    float ur = b[-BW + 1];
    float ll = b[+BW - 1];
    float lr = b[+BW + 1];
    float lx = 3.0f * (lr - ll + ur - ul) + 10.0f * (b[+1] - b[-1]);
    float ly = 3.0f * (lr + ll - ur - ul) + 10.0f * (b[BW] - b[-BW]);
    float dif2 = invk * (lx * lx + ly * ly);
    if (type == PM_G1)
      flowd[y * pitch + x] = exp(-dif2);
    else if (type == PM_G2)
      flowd[y * pitch + x] = 1.0f / (1.0f + dif2);
    else if (type == WEICKERT)
      flowd[y * pitch + x] = 1.0f - exp(-3.315 / (dif2 * dif2 * dif2 * dif2));
    else
      flowd[y * pitch + x] = 1.0f / sqrt(1.0f + dif2);
  }
}

double Flow(CudaImage &img, CudaImage &flow, DIFFUSIVITY_TYPE type,
            float kcontrast) {
  // TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, SCHARR_W), iDivUp(img.height, SCHARR_H));
  dim3 threads(SCHARR_W + 2, SCHARR_H + 2);
  Flow << <blocks, threads>>> (img.d_data, flow.d_data, img.width, img.pitch,
                               img.height, type,
                               1.0f / (kcontrast * kcontrast));
  // checkMsg("Flow() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // = timer0.read();
#ifdef VERBOSE
  printf("Flow time =                   %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

struct NLDStep_t {
  float *imgd;
  float *flod;
  float *temd;
  int width;
  int pitch;
  int height;
  float stepsize;
};

//__global__ void NLDStep(float *imgd, float *flod, float *temd, int width, int
// pitch, int height, float stepsize)
__global__ void NLDStep(NLDStep_t s) {
#undef BW
#define BW (NLDSTEP_W + 2)
  __shared__ float ibuff[BW * (NLDSTEP_H + 2)];
  __shared__ float fbuff[BW * (NLDSTEP_H + 2)];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * NLDSTEP_W + tx;
  int y = blockIdx.y * NLDSTEP_H + ty;
  int xp = (x == 0 ? 0 : (x > s.width ? s.width - 1 : x - 1));
  int yp = (y == 0 ? 0 : (y > s.height ? s.height - 1 : y - 1));
  ibuff[ty * BW + tx] = s.imgd[yp * s.pitch + xp];
  fbuff[ty * BW + tx] = s.flod[yp * s.pitch + xp];
  __syncthreads();
  if (tx < NLDSTEP_W && ty < NLDSTEP_H && x < s.width && y < s.height) {
    float *ib = ibuff + (ty + 1) * BW + (tx + 1);
    float *fb = fbuff + (ty + 1) * BW + (tx + 1);
    float ib0 = ib[0];
    float fb0 = fb[0];
    float xpos = (fb0 + fb[+1]) * (ib[+1] - ib0);
    float xneg = (fb0 + fb[-1]) * (ib0 - ib[-1]);
    float ypos = (fb0 + fb[+BW]) * (ib[+BW] - ib0);
    float yneg = (fb0 + fb[-BW]) * (ib0 - ib[-BW]);
    s.temd[y * s.pitch + x] = s.stepsize * (xpos - xneg + ypos - yneg);
  }
}

struct NLDUpdate_t {
  float *imgd;
  float *temd;
  int width;
  int pitch;
  int height;
};

//__global__ void NLDUpdate(float *imgd, float *temd, int width, int pitch, int
// height)
__global__ void NLDUpdate(NLDUpdate_t s) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x < s.width && y < s.height) {
    int p = y * s.pitch + x;
    s.imgd[p] = s.imgd[p] + s.temd[p];
  }
}

double NLDStep(CudaImage &img, CudaImage &flow, CudaImage &temp,
               float stepsize) {
  // TimerGPU timer0(0);
  dim3 blocks0(iDivUp(img.width, NLDSTEP_W), iDivUp(img.height, NLDSTEP_H));
  dim3 threads0(NLDSTEP_W + 2, NLDSTEP_H + 2);
  NLDStep_t s;
  s.imgd = img.d_data;
  s.flod = flow.d_data;
  s.temd = temp.d_data;
  s.width = img.width;
  s.pitch = img.pitch;
  s.height = img.height;
  s.stepsize = 0.5 * stepsize;
  // NLDStep<<<blocks0, threads0>>>(img.d_data, flow.d_data, temp.d_data,
  // img.width, img.pitch, img.height, 0.5f*stepsize);
  NLDStep << <blocks0, threads0>>> (s);
  // checkMsg("NLDStep() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  dim3 blocks1(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads1(32, 16);
  NLDUpdate_t su;
  su.imgd = img.d_data;
  su.temd = temp.d_data;
  su.width = img.width;
  su.height = img.height;
  su.pitch = img.pitch;
  // NLDUpdate<<<blocks1, threads1>>>(img.d_data, temp.d_data, img.width,
  // img.pitch, img.height);
  NLDUpdate << <blocks1, threads1>>> (su);
  // checkMsg("NLDUpdate() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // = timer0.read();
#ifdef VERBOSE
  printf("NLDStep time =                %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void HalfSample(float *iimd, float *oimd, int iwidth, int iheight,
                           int ipitch, int owidth, int oheight, int opitch) {
  __shared__ float buffer[16 * 33];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * 16 + tx;
  int y = blockIdx.y * 16 + ty;
  if (x >= owidth || y >= oheight) return;
  float *ptri = iimd + (2 * y) * ipitch + (2 * x);
  if (2 * owidth == iwidth) {
    buffer[ty * 32 + tx] = owidth * (ptri[0] + ptri[1]);
    ptri += ipitch;
    buffer[ty * 32 + tx + 16] = owidth * (ptri[0] + ptri[1]);
    if (ty == 15) {
      ptri += ipitch;
      buffer[tx + 32 * 16] = owidth * (ptri[0] + ptri[1]);
    } else if (y * 2 + 3 == iheight) {
      ptri += ipitch;
      buffer[tx + 32 * (ty + 1)] = owidth * (ptri[0] + ptri[1]);
    }
  } else {
    float f0 = owidth - x;
    float f2 = 1 + x;
    buffer[ty * 32 + tx] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[2];
    ptri += ipitch;
    buffer[ty * 32 + tx + 16] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[2];
    if (ty == 15 && 2 * oheight != iheight) {
      ptri += ipitch;
      buffer[tx + 32 * 16] = f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[1];
    } else if (y * 2 + 3 == iheight && 2 * oheight != iheight) {
      ptri += ipitch;
      buffer[tx + 32 * (ty + 1)] =
          f0 * ptri[0] + owidth * ptri[1] + f2 * ptri[2];
    }
  }
  __syncthreads();
  float *buff = buffer + 32 * ty + tx;
  if (2 * oheight == iheight)
    oimd[y * opitch + x] = oheight * (buff[0] + buff[16]) / (iwidth * iheight);
  else {
    float f0 = oheight - y;
    float f2 = 1 + y;
    oimd[y * opitch + x] = (f0 * buff[0] + oheight * buff[16] + f2 * buff[32]) /
                           (iwidth * iheight);
  }
}

__global__ void HalfSample2(float *iimd, float *oimd, int ipitch, int owidth,
                            int oheight, int opitch) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= owidth || y >= oheight) return;
  float *ptr = iimd + (2 * y) * ipitch + (2 * x);
  oimd[y * opitch + x] =
      0.25f * (ptr[0] + ptr[1] + ptr[ipitch + 0] + ptr[ipitch + 1]);
}

double HalfSample(CudaImage &inimg, CudaImage &outimg) {
  // TimerGPU timer0(0);
  if (inimg.width == 2 * outimg.width && inimg.height == 2 * outimg.height) {
    dim3 blocks(iDivUp(outimg.width, 32), iDivUp(outimg.height, 16));
    dim3 threads(32, 16);
    HalfSample2 << <blocks, threads>>> (inimg.d_data, outimg.d_data,
                                        inimg.pitch, outimg.width,
                                        outimg.height, outimg.pitch);
  } else {
    dim3 blocks(iDivUp(outimg.width, 16), iDivUp(outimg.height, 16));
    dim3 threads(16, 16);
    HalfSample << <blocks, threads>>> (inimg.d_data, outimg.d_data, inimg.width,
                                       inimg.height, inimg.pitch, outimg.width,
                                       outimg.height, outimg.pitch);
  }
  // checkMsg("HalfSample() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("HalfSample time =             %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double Copy(CudaImage &inimg, CudaImage &outimg) {
  // TimerGPU timer0(0);
  double gpuTime = 0;  // timer0.read();
  safeCall(cudaMemcpy2DAsync(outimg.d_data, sizeof(float) * outimg.pitch,
                             inimg.d_data, sizeof(float) * outimg.pitch,
                             sizeof(float) * inimg.width, inimg.height,
                             cudaMemcpyDeviceToDevice));
#ifdef VERBOSE
  printf("Copy time =                   %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

float *AllocBuffers(int width, int height, int num, int omax, int &maxpts,
                    std::vector<CudaImage> &buffers, cv::KeyPoint *&pts,
                    cv::KeyPoint *&ptsbuffer, int *&ptindices, unsigned char *&desc, float *&descbuffer, CudaImage *&ims) {

  maxpts = 4 * ((maxpts+3)/4);

  buffers.resize(omax * num);
  int w = width;
  int h = height;
  int p = iAlignUp(w, 128);
  int size = 0;
  for (int i = 0; i < omax; i++) {
    for (int j = 0; j < num; j++) {
      CudaImage &buf = buffers[i * num + j];
      buf.width = w;
      buf.height = h;
      buf.pitch = p;
      buf.d_data = (float *)((long)size);
      size += h * p;
    }
    w /= 2;
    h /= 2;
    p = iAlignUp(w, 128);
  }
  int ptsstart = size;
  size += sizeof(cv::KeyPoint) * maxpts / sizeof(float);
  int ptsbufferstart = size;
  size += sizeof(cv::KeyPoint) * maxpts / sizeof(float);
  int descstart = size;
  size += sizeof(unsigned char)*maxpts*61/sizeof(float);
  int descbufferstart = size;
  size += sizeof(float)*3*29*maxpts / sizeof(float);
  int indicesstart = size;
  size += 21*21*sizeof(int)*maxpts/sizeof(float);
  int imgstart = size;
  size += sizeof(CudaImage) * (num * omax + sizeof(float) - 1) / sizeof(float);
  float *memory = NULL;
  size_t pitch;

  std::cout << "allocating " << size/1024./1024. << " Mbytes of gpu memory\n";

  safeCall(cudaMallocPitch((void **)&memory, &pitch, (size_t)4096,
                           (size + 4095) / 4096 * sizeof(float)));
  for (int i = 0; i < omax * num; i++) {
    CudaImage &buf = buffers[i];
    buf.d_data = memory + (long)buf.d_data;
  }
  pts = (cv::KeyPoint *)(memory + ptsstart);
  ptsbuffer = (cv::KeyPoint *)(memory + ptsbufferstart);
  desc = (unsigned char *)(memory + descstart);
  descbuffer = (float*)(memory + descbufferstart);
  ptindices = (int*)(memory + indicesstart);
  ims = (CudaImage *)(memory + imgstart);

  InitCompareIndices();

  cudaStreamCreate(&copyStream);

  return memory;
}


void FreeBuffers(float *buffers) { safeCall(cudaFree(buffers)); }

__device__ unsigned int d_Maxval[1];
__device__ int d_Histogram[512];

#define CONTRAST_W 64
#define CONTRAST_H 7
#define HISTCONT_W 64
#define HISTCONT_H 8
#define HISTCONT_R 4

__global__ void MaxContrast(float *imgd, float *cond, int width, int pitch,
                            int height) {
#define WID (CONTRAST_W + 2)
  __shared__ float buffer[WID * (CONTRAST_H + 2)];
  __shared__ unsigned int maxval[32];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  if (tx < 32 && !ty) maxval[tx] = 0.0f;
  __syncthreads();
  int x = blockIdx.x * CONTRAST_W + tx;
  int y = blockIdx.y * CONTRAST_H + ty;
  if (x >= width || y >= height) return;
  float *b = buffer + ty * WID + tx;
  b[0] = imgd[y * pitch + x];
  __syncthreads();
  if (tx < CONTRAST_W && ty < CONTRAST_H && x < width - 2 && y < height - 2) {
    float dx = 3.0f * (b[0] - b[2] + b[2 * WID] - b[2 * WID + 2]) +
               10.0f * (b[WID] - b[WID + 2]);
    float dy = 3.0f * (b[0] + b[2] - b[2 * WID] - b[2 * WID + 2]) +
               10.0f * (b[1] - b[2 * WID + 1]);
    float grad = sqrt(dx * dx + dy * dy);
    cond[(y + 1) * pitch + (x + 1)] = grad;
    unsigned int *gradi = (unsigned int *)&grad;
    atomicMax(maxval + (tx & 31), *gradi);
  }
  __syncthreads();
  if (tx < 32 && !ty) atomicMax(d_Maxval, maxval[tx]);
}

__global__ void HistContrast(float *cond, int width, int pitch, int height,
                             float imaxval, int nbins) {
  __shared__ int hist[512];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = ty * HISTCONT_W + tx;
  if (i < nbins) hist[i] = 0;
  __syncthreads();
  int x = blockIdx.x * HISTCONT_W + tx;
  int y = blockIdx.y * HISTCONT_H * HISTCONT_R + ty;
  if (x > 0 && x < width - 1) {
    for (int i = 0; i < HISTCONT_R; i++) {
      if (y > 0 && y < height - 1) {
        int idx = min((int)(nbins * cond[y * pitch + x] * imaxval), nbins - 1);
        atomicAdd(hist + idx, 1);
      }
      y += HISTCONT_H;
    }
  }
  __syncthreads();
  if (i < nbins && hist[i] > 0) atomicAdd(d_Histogram + i, hist[i]);
}

double ContrastPercentile(CudaImage &img, CudaImage &temp, CudaImage &blur,
                          float perc, int nbins, float &contrast) {
  // TimerGPU timer0(0);
  LowPass(img, blur, temp, 1.0f, 5);

  float h_Maxval = 0.0f;
  safeCall(cudaMemcpyToSymbolAsync(d_Maxval, &h_Maxval, sizeof(float)));
  dim3 blocks1(iDivUp(img.width, CONTRAST_W), iDivUp(img.height, CONTRAST_H));
  dim3 threads1(CONTRAST_W + 2, CONTRAST_H + 2);
  MaxContrast << <blocks1, threads1>>>
      (blur.d_data, temp.d_data, blur.width, blur.pitch, blur.height);
  // checkMsg("MaxContrast() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  safeCall(cudaMemcpyFromSymbolAsync(&h_Maxval, d_Maxval, sizeof(float)));

  if (nbins > 512) {
    printf(
        "Warning: Largest number of possible bins in ContrastPercentile() is "
        "512\n");
    nbins = 512;
  }
  int h_Histogram[512];
  memset(h_Histogram, 0, nbins * sizeof(int));
  safeCall(
      cudaMemcpyToSymbolAsync(d_Histogram, h_Histogram, nbins * sizeof(int)));
  dim3 blocks2(iDivUp(temp.width, HISTCONT_W),
               iDivUp(temp.height, HISTCONT_H * HISTCONT_R));
  dim3 threads2(HISTCONT_W, HISTCONT_H);
  HistContrast << <blocks2, threads2>>> (temp.d_data, temp.width, temp.pitch,
                                         temp.height, 1.0f / h_Maxval, nbins);
  safeCall(
      cudaMemcpyFromSymbolAsync(h_Histogram, d_Histogram, nbins * sizeof(int)));

  int npoints = (temp.width - 2) * (temp.height - 2);
  int nthreshold = (int)(npoints * perc);
  int k = 0, nelements = 0;
  for (k = 0; nelements < nthreshold && k < nbins; k++)
    nelements += h_Histogram[k];
  contrast = (nelements < nthreshold ? 0.03f : h_Maxval * ((float)k / nbins));

  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("ContrastPercentile time =     %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void Derivate(float *imd, float *lxd, float *lyd, int width,
                         int pitch, int height, int step, float fac1,
                         float fac2) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= width || y >= height) return;
  int xl = (x < step ? step - x : x - step);
  int xh = (x >= width - step ? 2 * width - x - step - 2 : x + step);
  int yl = (y < step ? step - y : y - step);
  int yh = (y >= height - step ? 2 * height - y - step - 2 : y + step);
  float ul = imd[yl * pitch + xl];
  float ur = imd[yl * pitch + xh];
  float ll = imd[yh * pitch + xl];
  float lr = imd[yh * pitch + xh];
  float cl = imd[y * pitch + xl];
  float cr = imd[y * pitch + xh];
  lxd[y * pitch + x] = fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl);
  float uc = imd[yl * pitch + x];
  float lc = imd[yh * pitch + x];
  lyd[y * pitch + x] = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
}

__global__ void HessianDeterminant(float *lxd, float *lyd, float *detd,
                                   int width, int pitch, int height, int step,
                                   float fac1, float fac2) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x >= width || y >= height) return;
  int xl = (x < step ? step - x : x - step);
  int xh = (x >= width - step ? 2 * width - x - step - 2 : x + step);
  int yl = (y < step ? step - y : y - step);
  int yh = (y >= height - step ? 2 * height - y - step - 2 : y + step);
  float ul = lxd[yl * pitch + xl];
  float ur = lxd[yl * pitch + xh];
  float ll = lxd[yh * pitch + xl];
  float lr = lxd[yh * pitch + xh];
  float cl = lxd[y * pitch + xl];
  float cr = lxd[y * pitch + xh];
  float lxx = fac1 * (ur + lr - ul - ll) + fac2 * (cr - cl);
  float uc = lxd[yl * pitch + x];
  float lc = lxd[yh * pitch + x];
  float lyx = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
  ul = lyd[yl * pitch + xl];
  ur = lyd[yl * pitch + xh];
  ll = lyd[yh * pitch + xl];
  lr = lyd[yh * pitch + xh];
  uc = lyd[yl * pitch + x];
  lc = lyd[yh * pitch + x];
  float lyy = fac1 * (lr + ll - ur - ul) + fac2 * (lc - uc);
  detd[y * pitch + x] = lxx * lyy - lyx * lyx;
}

double HessianDeterminant(CudaImage &img, CudaImage &lx, CudaImage &ly,
                          int step) {
  // TimerGPU timer0(0);
  float w = 10.0 / 3.0;
  float fac1 = 1.0 / (2.0 * (w + 2.0));
  float fac2 = w * fac1;
  dim3 blocks(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads(32, 16);
  Derivate << <blocks, threads>>> (img.d_data, lx.d_data, ly.d_data, img.width,
                                   img.pitch, img.height, step, fac1, fac2);
  // checkMsg("Derivate() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  HessianDeterminant << <blocks, threads>>> (lx.d_data, ly.d_data, img.d_data,
                                             img.width, img.pitch, img.height,
                                             step, fac1, fac2);
  // checkMsg("HessianDeterminant() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("HessianDeterminant time =     %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

__global__ void FindExtrema(float *imd, float *imp, float *imn, int maxx,
                            int pitch, int maxy, float border, float dthreshold,
                            int scale, int octave, float size,
                            cv::KeyPoint *pts, int maxpts) {
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;

  int left_x = (int)(x - border + 0.5f) - 1;
  int right_x = (int)(x + border + 0.5f) + 1;
  int up_y = (int)(y - border + 0.5f) - 1;
  int down_y = (int)(y + border + 0.5f) + 1;
  if (left_x < 0 || right_x >= maxx || up_y < 0 || down_y >= maxy) return;
  int p = y * pitch + x;
  float v = imd[p];
  if (v > dthreshold && v > imd[p - pitch - 1] && v > imd[p + pitch + 1] &&
      v > imd[p + pitch - 1] && v > imd[p - pitch + 1] && v > imd[p - 1] &&
      v > imd[p + 1] && v > imd[p + pitch] && v > imd[p - pitch]) {
    float dx = 0.5f * (imd[p + 1] - imd[p - 1]);
    float dy = 0.5f * (imd[p + pitch] - imd[p - pitch]);
    float dxx = imd[p + 1] + imd[p - 1] - 2.0f * v;
    float dyy = imd[p + pitch] + imd[p - pitch] - 2.0f * v;
    float dxy = 0.25f * (imd[p + pitch + 1] + imd[p - pitch - 1] -
                         imd[p + pitch - 1] - imd[p - pitch + 1]);
    float det = dxx * dyy - dxy * dxy;
    float idet = (det != 0.0f ? 1.0f / det : 0.0f);
    float dst0 = idet * (dxy * dy - dyy * dx);
    float dst1 = idet * (dxy * dx - dxx * dy);
    bool weak = true;
    if (dst0 >= -1.0f && dst0 <= 1.0f && dst1 >= -1.0f && dst1 <= 1.0f) {
      weak = 0;
    }
    unsigned int idx = atomicInc(d_PointCounter, 0x7fffffff);
    if (idx < maxpts) {
      cv::KeyPoint &point = pts[idx];
      point.response = v;
      point.size = (weak ? -1 : 1) * 2.0 * size;
      float octsub = (dst0 < 0 ? -1 : 1) * (octave + fabs(dst0));
      *(float *)(&point.octave) = (weak ? octave : octsub);
      point.class_id = scale;
      int ratio = (1 << octave);
      point.pt.x = ratio * (x);
      point.pt.y = ratio * (y);
      point.angle = dst1;
    } else {
        atomicAdd(d_PointCounter,-1);
    }
  }
}

__global__ void CopyIdxArray(int scale) {
  d_ExtremaIdx[scale] = d_PointCounter[0];
}

double FindExtrema(CudaImage &img, CudaImage &imgp, CudaImage &imgn,
                   float border, float dthreshold, int scale, int octave,
                   float size, cv::KeyPoint *pts, int maxpts) {
  // TimerGPU timer0(0);
  dim3 blocks(iDivUp(img.width, 32), iDivUp(img.height, 16));
  dim3 threads(32, 16);
  float b = border;
  FindExtrema << <blocks, threads>>>
      (img.d_data, imgp.d_data, imgn.d_data, img.width, img.pitch, img.height,
       b, dthreshold, scale, octave, size, pts, maxpts);

  CopyIdxArray << <1, 1>>> (scale);
CHK

  // checkMsg("FindExtrema() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("FindExtrema time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

void ClearPoints() {
  int totPts = 0;
  safeCall(cudaMemcpyToSymbolAsync(d_PointCounter, &totPts, sizeof(int)));
}

__forceinline__ __device__ void atomicSort(int *pts, int shmidx, int offset,
                                           int sortdir) {
  int &p0 = pts[shmidx + sortdir];
  int &p1 = pts[shmidx + (offset - sortdir)];

  if (p0 < p1) {
    int t = p0;
    p0 = p1;
    p1 = t;
  }
}

__forceinline__ __device__ bool atomicCompare(const cv::KeyPoint &i,
                                              const cv::KeyPoint &j) {
  float t = i.pt.x * j.pt.x;
  if (t == 0) {
    if (j.pt.x != 0) {
      return false;
    } else {
      return true;
    }
  }

  if (i.pt.y < j.pt.y) return true;
  if (i.pt.y == j.pt.y && i.pt.x < j.pt.x) return true;

  return false;
}

template <typename T>
struct sortstruct_t {
    T idx;
    short x;
    short y;
};

template <typename T>
__forceinline__ __device__ bool atomicCompare(const sortstruct_t<T> &i,
                                              const sortstruct_t<T> &j) {
    int t = i.x * j.x;
    if (t == 0) {
	if (j.x != 0) {
	    return false;
	} else {
	    return true;
	}
    }

    if (i.y < j.y) return true;

    if (i.y == j.y && i.x < j.x) return true;

  return false;
}

template <typename T>
__forceinline__ __device__ void atomicSort(sortstruct_t<T> *pts, int shmidx,
                                           int offset, int sortdir) {
    sortstruct_t<T> &p0 = pts[(shmidx + sortdir)];
    sortstruct_t<T> &p1 = pts[(shmidx + (offset - sortdir))];

  if (atomicCompare(p0, p1)) {
      int idx = p0.idx;
      short ptx = p0.x;
      short pty = p0.y;
      p0.idx = p1.idx;
      p0.x = p1.x;
      p0.y = p1.y;
      p1.idx = idx;
      p1.x = ptx;
      p1.y = pty;
  }
}

#define BitonicSortThreads 1024
template <class T>
__global__ void bitonicSort(const T *pts, T *newpts) {
  int scale = blockIdx.x;

  __shared__ struct sortstruct_t<short> shm[8192];

  int first = scale == 0 ? 0 : d_ExtremaIdx[scale - 1];
  int last = d_ExtremaIdx[scale];

  int nkpts = last - first;

  const cv::KeyPoint *tmpg = &pts[first];

  for (int i = threadIdx.x; i < 8192;
       i += BitonicSortThreads) {
    if (i < nkpts) {
      shm[i].idx = i;
      shm[i].y = (short)tmpg[i].pt.y;
      shm[i].x = (short)tmpg[i].pt.x;
    } else {
      shm[i].idx = -1;
      shm[i].y = 0;
      shm[i].x = 0;
    }
  }
  __syncthreads();


  for (int i=1; i<8192; i <<= 1) {
      for (int j=i; j>0; j >>= 1) {
	  int tx = threadIdx.x;
	  int mask = 0x0fffffff * j;
	  for (int idx=0; idx<4096; idx+=BitonicSortThreads) {
	      int sortdir = (tx & i) > 0 ? 0 : 1;
	      int tidx = ((tx & mask) << 1) + (tx & ~mask);
	      atomicSort(shm, tidx, j, j*sortdir);
	      tx += BitonicSortThreads;
	      __syncthreads();
	  }
      }
  }
  

  cv::KeyPoint *tmpnewg = &newpts[first];
  for (int i = 0; i < 8192; i += BitonicSortThreads) {
    if (i + threadIdx.x < nkpts) {
      tmpnewg[i + threadIdx.x].angle = tmpg[shm[i + threadIdx.x].idx].angle;
      tmpnewg[i + threadIdx.x].class_id = tmpg[shm[i + threadIdx.x].idx].class_id;
      tmpnewg[i + threadIdx.x].octave = tmpg[shm[i + threadIdx.x].idx].octave;
      tmpnewg[i + threadIdx.x].pt.y = tmpg[shm[i + threadIdx.x].idx].pt.y;
      tmpnewg[i + threadIdx.x].pt.x = tmpg[shm[i + threadIdx.x].idx].pt.x;
      tmpnewg[i + threadIdx.x].response =
          tmpg[shm[i + threadIdx.x].idx].response;
      tmpnewg[i + threadIdx.x].size = tmpg[shm[i + threadIdx.x].idx].size;
    }
  }
}

template <class T>
__global__ void bitonicSort_global(const T *pts, T *newpts, sortstruct_t<int>* _shm, int _sz) {
  int scale = blockIdx.x;

  //__shared__ struct sortstruct_t shm[8192];

  int first = scale == 0 ? 0 : d_ExtremaIdx[scale - 1];
  int last = d_ExtremaIdx[scale];

  int nkpts = last - first;

  const cv::KeyPoint *tmpg = &pts[first];

  int nkpts_ceil = 1;
  while (nkpts_ceil < nkpts) nkpts_ceil *= 2;

  sortstruct_t<int> *shm = &(_shm[_sz*blockIdx.x]);
  
  for (int i = threadIdx.x; i < nkpts_ceil;
       i += BitonicSortThreads) {
    if (i < nkpts) {
      shm[i].idx = i;
      shm[i].y = (short)tmpg[i].pt.y;
      shm[i].x = (short)tmpg[i].pt.x;
    } else {
      shm[i].idx = -1;
      shm[i].y = 0;
      shm[i].x = 0;
    }
  }
  __syncthreads();


  for (int i=1; i<nkpts_ceil; i <<= 1) {
      for (int j=i; j>0; j >>= 1) {
	  int tx = threadIdx.x;
	  int mask = 0x0fffffff * j;
	  for (int idx=0; idx<nkpts_ceil/2; idx+=BitonicSortThreads) {
	      int sortdir = (tx & i) > 0 ? 0 : 1;
	      int tidx = ((tx & mask) << 1) + (tx & ~mask);
	      atomicSort(shm, tidx, j, j*sortdir);
	      tx += BitonicSortThreads;
	      __syncthreads();
	  }
      }
  }
  

  cv::KeyPoint *tmpnewg = &newpts[first];
  for (int i = 0; i < nkpts_ceil; i += BitonicSortThreads) {
    if (i + threadIdx.x < nkpts) {
      tmpnewg[i + threadIdx.x].angle = tmpg[shm[i + threadIdx.x].idx].angle;
      tmpnewg[i + threadIdx.x].class_id = tmpg[shm[i + threadIdx.x].idx].class_id;
      tmpnewg[i + threadIdx.x].octave = tmpg[shm[i + threadIdx.x].idx].octave;
      tmpnewg[i + threadIdx.x].pt.y = tmpg[shm[i + threadIdx.x].idx].pt.y;
      tmpnewg[i + threadIdx.x].pt.x = tmpg[shm[i + threadIdx.x].idx].pt.x;
      tmpnewg[i + threadIdx.x].response =
          tmpg[shm[i + threadIdx.x].idx].response;
      tmpnewg[i + threadIdx.x].size = tmpg[shm[i + threadIdx.x].idx].size;
    }
  }
}


#define FindNeighborsThreads 32
__global__ void FindNeighbors(cv::KeyPoint *pts, int *kptindices, int width) {
  __shared__ int gidx[1];

  // which scale?
  int scale = pts[blockIdx.x].class_id;

  int cmpIdx = scale < 1 ? 0 : d_ExtremaIdx[scale - 1];

  float size = pts[blockIdx.x].size;

  gidx[0] = 1;
  __syncthreads();

  // One keypoint per block.
  cv::KeyPoint &kpt = pts[blockIdx.x];

  // Key point to compare. Only compare with smaller than current
  // Iterate backwards instead and break as soon as possible!
  //for (int i = cmpIdx + threadIdx.x; i < blockIdx.x; i += FindNeighborsThreads) {
  for (int i=blockIdx.x-threadIdx.x-1; i >= cmpIdx; i -= FindNeighborsThreads) {
      
      cv::KeyPoint &kpt_cmp = pts[i];
      
      if (kpt.pt.y-kpt_cmp.pt.y > size*.5f) break;
      
      //if (fabs(kpt.pt.y-kpt_cmp.pt.y) > size*.5f) continue;
      
      float dist = (kpt.pt.x - kpt_cmp.pt.x) * (kpt.pt.x - kpt_cmp.pt.x) +
	  (kpt.pt.y - kpt_cmp.pt.y) * (kpt.pt.y - kpt_cmp.pt.y);
      
      if (dist < size * size * 0.25) {
	  int idx = atomicAdd(gidx, 1);
	  kptindices[blockIdx.x * width + idx] = i;
      }
  }

  if (scale > 0) {
      int startidx = d_ExtremaIdx[scale-1];
      cmpIdx = scale < 2 ? 0 : d_ExtremaIdx[scale - 2];
      for (int i=startidx-threadIdx.x-1; i >= cmpIdx; i -= FindNeighborsThreads) {	  
	  cv::KeyPoint &kpt_cmp = pts[i];
	  
	  if (kpt_cmp.pt.y-kpt.pt.y > size*.5f) continue;
	  
	  if (kpt.pt.y-kpt_cmp.pt.y > size*.5f) break;
	  
	  float dist = (kpt.pt.x - kpt_cmp.pt.x) * (kpt.pt.x - kpt_cmp.pt.x) +
	      (kpt.pt.y - kpt_cmp.pt.y) * (kpt.pt.y - kpt_cmp.pt.y);
	  
	  if (dist < size * size * 0.25) {
	      int idx = atomicAdd(gidx, 1);
	      kptindices[blockIdx.x * width + idx] = i;
	  }
      }
  }

      


  __syncthreads();

  if (threadIdx.x == 0) {
    kptindices[blockIdx.x * width] = gidx[0];
  }
}

// TODO Intermediate storage of memberarray and minneighbor
#define FilterExtremaThreads 1024
__global__ void FilterExtrema_kernel(cv::KeyPoint *kpts, cv::KeyPoint *newkpts,
				     int *kptindices, int width,
				     int *memberarray,
				     int *minneighbor,
				     char  *shouldAdd) {
  // -1  means not processed
  // -2  means added but replaced
  // >=0 means added


    __shared__ bool shouldBreak[1];

  int nump = d_PointCounter[0];

  // Initially all points are unprocessed
  for (int i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
      memberarray[i] = -1;
  }

  if (threadIdx.x == 0) {
    shouldBreak[0] = true;
  }

  __syncthreads();

  // Loop until there are no more points to process
  for (int xx=0; xx<10000; ++xx) {
      //while (true) {

      // Outer loop to handle more than 8*1024 points
      // Start by restoring memberarray
      // Make sure to add appropriate offset to indices
      // for (int offset=0; offset<nump; offset += 8*1024) {
        // memberarray[i] = storedmemberarray[i+offset];

      //for (int offset=0; offset<nump; offset += 8*1024) {

	  // Mark all points for addition and no minimum neighbor
      //int maxi = nump-offset >= 8*1024 ? 8*1024 : nump-offset;
      for (size_t i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
	  minneighbor[i] = nump+1;
	  shouldAdd[i] = true;
      }
      __syncthreads();

    // Look through all points. If there are points that have not been processed,
    // disable breaking and check if it has no processed neighbors (add), has all processed
    // neighbors (compare with neighbors) or has some unprocessed neighbor (wait)
    for (size_t i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
      int neighborsSize = kptindices[i * width] - 1;
      int *neighbors = &(kptindices[i * width + 1]);

      // Only do if we didn't process the point before
      if (memberarray[i] == -1) {
        // If we process at least one point we shouldn't break
        // No need to sync. Only want to know if at least one thread wants to
        // continue
        shouldBreak[0] = false;
        // Sort neighbors according to the order of currently added points
        // (often very few)
        // If the neighbor has been replaced, stick it to the back
        // If any neighbor has not been processed, break;
        bool shouldProcess = true;
        for (int k = 0; k < neighborsSize; ++k) {
          // If the point has one or more unprocessed neighbors, skip
          if (memberarray[neighbors[k]] == -1) {
            shouldProcess = false;
            shouldAdd[i] = false;
            break;
          }
          // If it has a neighbor that is in the list, we don't add, but process
          if (memberarray[neighbors[k]] >= 0) {
            shouldAdd[i] = false;
          }
        }

        // We should process and potentially replace the neighbor
        if (shouldProcess && !shouldAdd[i]) {
          // Find the smallest neighbor. Often only one or two, so no ned for fancy algorithm
          for (int k = 0; k < neighborsSize; ++k) {
            for (int j = k + 1; j < neighborsSize; ++j) {
              if (memberarray[neighbors[k]] == -2 ||
                  (memberarray[neighbors[j]] != -2 &&
                   memberarray[neighbors[j]] < memberarray[neighbors[k]])) {
                int t = neighbors[k];
                neighbors[k] = neighbors[j];
                neighbors[j] = t;
              }
            }
          }
          // Pick the first neighbor
          // We need to make sure, in case more than one point has this
          // neighbor,
          // That the point with lowest memberarrayindex processes it first
          // Here minneighbor[i] is the target and i the neighbor
          int nidx = neighbors[0];
          minneighbor[nidx] = min(minneighbor[nidx], (int)i);
        }
      }
    }
    __syncthreads();

    // Check which points we can add
    for (size_t i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
      if (memberarray[i] == -1) {
        if (shouldAdd[i]) {
          memberarray[i] = i;
        }
      }
    }
    __syncthreads();

    // Look at the neighbors. If the response is higher, replace
    for (size_t i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
      if (minneighbor[i] != nump+1) {
        if (memberarray[minneighbor[i]] == -1) {
          if (!shouldAdd[minneighbor[i]]) {
            const cv::KeyPoint &p0 = kpts[minneighbor[i]];
            const cv::KeyPoint &p1 = kpts[i];
            if (p0.response > p1.response) {
              memberarray[minneighbor[i]] = i;
              memberarray[i] = -2;
            } else {
              memberarray[minneighbor[i]] = -2;
            }
          }
        }
      }
    }
    __syncthreads();

    // End outer loop
    //for (size_t i = threadIdx.x; i < nump; i += FilterExtremaThreads) {
//	storedmemberarray[i+offset] = memberarray[i];
    //  }
    // __syncthreads();
    //}

    // Are we done?
    if (shouldBreak[0]) break;

    if (threadIdx.x == 0) {
      shouldBreak[0] = true;
    }
    __syncthreads();

  }

  __syncthreads();

}


__global__ void sortFiltered_kernel(cv::KeyPoint *kpts, cv::KeyPoint *newkpts,
				    int *memberarray) {


    __shared__ int minneighbor[2048];
  __shared__ int curridx[1];

  int nump = d_PointCounter[0];

  if (threadIdx.x == 0) {
    curridx[0] = 0;
  }

// Sort array
  const int upper = (nump + 2047) & (0xfffff800);

  for (int i = threadIdx.x; i < upper; i += 2 * FilterExtremaThreads) {

    minneighbor[threadIdx.x] =
        i >= nump ? nump+1 : (memberarray[i] < 0 ? nump+1 : (kpts[memberarray[i]].size < 0 ? nump+1 : memberarray[i]));
    minneighbor[threadIdx.x + 1024] =
        i + 1024 >= nump ? nump+1
                         : (memberarray[i + 1024] < 0 ? nump+1 : (kpts[memberarray[i+1024]].size < 0 ? nump+1 : memberarray[i+1024]));

    __syncthreads();

    // Sort and store keypoints
#pragma unroll 1
    for (int k = 1; k < 2048; k <<= 1) {
      int sortdir = (threadIdx.x & k) > 0 ? 0 : 1;

#pragma unroll 1
      for (int j = k; j > 0; j >>= 1) {
        int mask = 0x0fffffff * j;
        int tidx = ((threadIdx.x & mask) << 1) + (threadIdx.x & ~mask);
        atomicSort(minneighbor, tidx, j, j * sortdir);
        __syncthreads();
      }
    }

    __syncthreads();

#pragma unroll 1
    for (int k = threadIdx.x; k < 2048; k += 1024) {
      if (minneighbor[k] < nump) {
          // Restore subpixel component
	  cv::KeyPoint &okpt = kpts[minneighbor[k]];
          float octsub = fabs(*(float*)(&kpts[minneighbor[k]].octave));
          int octave = (int)octsub;
          float subp = (*(float*)(&kpts[minneighbor[k]].octave) < 0 ? -1 : 1) * (octsub - octave);
          float ratio = 1 << octave;
	  cv::KeyPoint &tkpt = newkpts[k + curridx[0]];
	  tkpt.pt.y = ratio * ((int)(0.5f+okpt.pt.y / ratio) + okpt.angle);
	  tkpt.pt.x = ratio * ((int)(0.5f+okpt.pt.x / ratio) + subp);
	  // newkpts[k + curridx[0] + threadIdx.x].angle = 0; // This will be set elsewhere
	  tkpt.class_id = okpt.class_id;
	  tkpt.octave = octave;
	  tkpt.response =  okpt.response;
	  tkpt.size = okpt.size;
      }
    }
    __syncthreads();


    // How many did we add?
    if (minneighbor[2047] < nump) {
      curridx[0] += 2048;
    } else {
      if (minneighbor[1024] < nump) {
        if (threadIdx.x < 1023 && minneighbor[1024 + threadIdx.x] < nump &&
            minneighbor[1024 + threadIdx.x + 1] == nump+1) {
          curridx[0] += 1024 + threadIdx.x + 1;
        }
      } else {
        if (minneighbor[threadIdx.x] < nump &&
            minneighbor[threadIdx.x + 1] == nump+1) {
          curridx[0] += threadIdx.x + 1;
        }
      }
      __syncthreads();
    }
    
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    d_PointCounter[0] = curridx[0];
  }
}

void FilterExtrema(cv::KeyPoint *pts, cv::KeyPoint *newpts, int* kptindices, int& nump) {

  //int nump;
  cudaMemcpyFromSymbol(&nump, d_PointCounter, sizeof(int));

  unsigned int extremaidx_h[16];
  cudaMemcpyFromSymbol(extremaidx_h,d_ExtremaIdx,16*sizeof(unsigned int));
  int maxnump = extremaidx_h[0];
  for (int i=1; i<16; ++i) {
      maxnump = max(maxnump,extremaidx_h[i]-extremaidx_h[i-1]);
  }

  int width = ceil(21) * ceil(21);

  // Sort the list of points
  dim3 blocks(16, 1, 1);
  dim3 threads(BitonicSortThreads, 1, 1);

  if (maxnump <= 8*1024) {
      bitonicSort << <blocks, threads>>> (pts, newpts);
  } else {
      int nump_ceil = 1;
      while (nump_ceil < nump) nump_ceil <<= 1;

      std::cout << "numpceil: " << nump_ceil << std::endl;
      
      sortstruct_t<int>* sortstruct;
      cudaMalloc((void**)&sortstruct, nump_ceil*16*sizeof(sortstruct_t<int>));
      bitonicSort_global << <blocks, threads>>> (pts, newpts, sortstruct,nump_ceil);
      cudaFree(sortstruct);
  }
CHK

  

  
/*  cv::KeyPoint* newpts_h = new cv::KeyPoint[nump];
  cudaMemcpy(newpts_h,newpts,nump*sizeof(cv::KeyPoint),cudaMemcpyDeviceToHost);

  int scale = 0;
  for (int i=1; i<nump; ++i) {
      cv::KeyPoint &k0 = newpts_h[i-1];
      cv::KeyPoint &k1 = newpts_h[i];

      std::cout << i << ": " << newpts_h[i].class_id << ": " << newpts_h[i].pt.y << " " << newpts_h[i].pt.x << ", " << newpts_h[i].size;

      if (!(k0.pt.y<k1.pt.y || (k0.pt.y==k1.pt.y && k0.pt.x<k1.pt.x))) std::cout << "  <<<<";
      if (k1.size < 0 ) std::cout << "  ##############";
       

      std::cout << "\n";

  }
*/

    // Find all neighbors
  cudaStreamSynchronize(copyStream);
  blocks.x = nump;
  threads.x = FindNeighborsThreads;
  FindNeighbors << <blocks, threads>>> (newpts, kptindices, width);
CHK
  //cudaDeviceSynchronize();
  //safeCall(cudaGetLastError());
  
  // Filter extrema
  blocks.x = 1;
  threads.x = FilterExtremaThreads;
  int *buffer1, *buffer2;
  cudaMalloc((void**)&buffer1, nump*sizeof(int));
  cudaMalloc((void**)&buffer2, nump*sizeof(int));
  char* buffer3;
  cudaMalloc((void**)&buffer3, nump);
  FilterExtrema_kernel << <blocks, threads>>> (newpts, pts, kptindices, width,
					       buffer1, buffer2, buffer3);
  threads.x = 1024;
  sortFiltered_kernel << <blocks, threads>>> (newpts, pts, buffer1);
CHK
  //cudaDeviceSynchronize();
  //safeCall(cudaGetLastError());
  cudaFree(buffer1);
  cudaFree(buffer2);
  cudaFree(buffer3);
  cudaMemcpyFromSymbolAsync(&nump, d_PointCounter, sizeof(int));

}


int GetPoints(std::vector<cv::KeyPoint> &h_pts, cv::KeyPoint *d_pts, int numPts) {
  h_pts.resize(numPts);
  safeCall(cudaMemcpyAsync((float *)&h_pts[0], d_pts,
                           sizeof(cv::KeyPoint) * numPts,
                           cudaMemcpyDeviceToHost, copyStream));
  return numPts;
}


void GetDescriptors(cv::Mat &h_desc, cv::Mat &d_desc, int numPts) {
    h_desc = cv::Mat(numPts, 61, CV_8U);
    cudaMemcpyAsync(h_desc.data, d_desc.data, numPts*61, cudaMemcpyDeviceToHost, copyStream);
}



__global__ void ExtractDescriptors(cv::KeyPoint *d_pts, CudaImage *d_imgs,
                                   float *_vals, int size2, int size3,
                                   int size4) {
  __shared__ float acc_vals[3 * 30 * EXTRACT_S];

  float *acc_vals_im = &acc_vals[0];
  float *acc_vals_dx = &acc_vals[30 * EXTRACT_S];
  float *acc_vals_dy = &acc_vals[2 * 30 * EXTRACT_S];

  int p = blockIdx.x;

  float *vals = &_vals[p * 3 * 29];

  float iratio = 1.0f / (1 << d_pts[p].octave);
  int scale = (int)(0.5f * d_pts[p].size * iratio + 0.5f);
  float xf = d_pts[p].pt.x * iratio;
  float yf = d_pts[p].pt.y * iratio;
  float ang = d_pts[p].angle;
  float co = cos(ang);
  float si = sin(ang);
  int tx = threadIdx.x;
  int lev = d_pts[p].class_id;
  float *imd = d_imgs[4 * lev + 0].d_data;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int winsize = max(3 * size3, 4 * size4);

  for (int i = 0; i < 30; ++i) {
    acc_vals_im[i * EXTRACT_S + tx] = 0.f;
    acc_vals_dx[i * EXTRACT_S + tx] = 0.f;
    acc_vals_dy[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float im = imd[pos];
    float dx = dxd[pos];
    float dy = dyd[pos];
    float rx = -dx * si + dy * co;
    float ry = dx * co + dy * si;

    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // Add 2x2
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx] += im;
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (y2 * 2 + x2) + 3 * 30 * tx + 2] += ry;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // Add 3x3
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx] += im;
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (4 + y3 * 3 + x3) + 3 * 30 * tx + 2] += ry;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // Add 4x4
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx] += im;
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx + 1] += rx;
      acc_vals[3 * (4 + 9 + y4 * 4 + x4) + 3 * 30 * tx + 2] += ry;
    }
  }

  __syncthreads();


// Reduce stuff
  float acc_reg;
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    for (int d = 0; d < 90; d += 30) {
      if (tx_d < 32) {
        acc_reg = acc_vals[3 * 30 * tx_d + offset + d] +
                  acc_vals[3 * 30 * (tx_d + 32) + offset + d];
        acc_reg += fake_shfl_down(acc_reg, 1);
        acc_reg += fake_shfl_down(acc_reg, 2);
        acc_reg += fake_shfl_down(acc_reg, 4);
        acc_reg += fake_shfl_down(acc_reg, 8);
        acc_reg += fake_shfl_down(acc_reg, 16);
      }
      if (tx_d == 0) {
        acc_vals[offset + d] = acc_reg;
      }
    }
  }

  __syncthreads();

  // Have 29*3 values to store
  // They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
  if (tx < 29) {
    vals[tx] = acc_vals[tx];
    vals[29 + tx] = acc_vals[29 + tx];
    vals[2 * 29 + tx] = acc_vals[2 * 29 + tx];
  }
}

__global__ void ExtractDescriptors_serial(cv::KeyPoint *d_pts,
                                          CudaImage *d_imgs, float *_vals,
                                          int size2, int size3, int size4) {
  __shared__ float acc_vals[30 * EXTRACT_S];
  __shared__ float final_vals[3 * 30];

  int p = blockIdx.x;

  float *vals = &_vals[p * 3 * 29];

  float iratio = 1.0f / (1 << d_pts[p].octave);
  int scale = (int)(0.5f * d_pts[p].size * iratio + 0.5f);
  float xf = d_pts[p].pt.x * iratio;
  float yf = d_pts[p].pt.y * iratio;
  float ang = d_pts[p].angle;
  float co = cos(ang);
  float si = sin(ang);
  int tx = threadIdx.x;
  int lev = d_pts[p].class_id;
  float *imd = d_imgs[4 * lev + 0].d_data;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int winsize = max(3 * size3, 4 * size4);

  // IM
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float im = imd[pos];
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += im;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += im;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += im;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  // DX
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float dx = dxd[pos];
    float dy = dyd[pos];
    float rx = -dx * si + dy * co;
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += rx;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += rx;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += rx;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  // DY
  for (int i = 0; i < 30; ++i) {
    acc_vals[i * EXTRACT_S + tx] = 0.f;
  }

  __syncthreads();

  for (int i = tx; i < winsize * winsize; i += EXTRACT_S) {
    int y = i / winsize;
    int x = i - winsize * y;
    int m = max(x, y);
    if (m >= winsize) continue;
    int l = x - size2;
    int k = y - size2;
    int xp = (int)(xf + scale * (k * co - l * si) + 0.5f);
    int yp = (int)(yf + scale * (k * si + l * co) + 0.5f);
    int pos = yp * pitch + xp;
    float dx = dxd[pos];
    float dy = dyd[pos];
    float ry = dx * co + dy * si;
    if (m < 2 * size2) {
      int x2 = (x < size2 ? 0 : 1);
      int y2 = (y < size2 ? 0 : 1);
      // atomicAdd(norm2, (x < size2 && y < size2 ? 1 : 0));
      // Add 2x2
      acc_vals[(y2 * 2 + x2) + 30 * tx] += ry;
    }
    if (m < 3 * size3) {
      int x3 = (x < size3 ? 0 : (x < 2 * size3 ? 1 : 2));
      int y3 = (y < size3 ? 0 : (y < 2 * size3 ? 1 : 2));
      // atomicAdd(norm3, (x < size3 && y < size3 ? 1 : 0));
      // Add 3x3
      acc_vals[(4 + y3 * 3 + x3) + 30 * tx] += ry;
    }
    if (m < 4 * size4) {
      int x4 = (x < 2 * size4 ? (x < size4 ? 0 : 1) : (x < 3 * size4 ? 2 : 3));
      int y4 = (y < 2 * size4 ? (y < size4 ? 0 : 1) : (y < 3 * size4 ? 2 : 3));
      // atomicAdd(norm4, (x < size4 && y < size4 ? 1 : 0));
      // Add 4x4
      acc_vals[(4 + 9 + y4 * 4 + x4) + 30 * tx] += ry;
    }
  }

  __syncthreads();

// Reduce stuff
#pragma unroll
  for (int i = 0; i < 15; ++i) {
    // 0..31 takes care of even accs, 32..63 takes care of odd accs
    int offset = 2 * i + (tx < 32 ? 0 : 1);
    int tx_d = tx < 32 ? tx : tx - 32;
    int acc_idx = 30 * tx_d + offset;
    if (tx_d < 32) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 32];
    }
    if (tx_d < 16) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 16];
    }
    if (tx_d < 8) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 8];
    }
    if (tx_d < 4) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 4];
    }
    if (tx_d < 2) {
      acc_vals[acc_idx] += acc_vals[acc_idx + 30 * 2];
    }
    if (tx_d < 1) {
      final_vals[3 * offset] = acc_vals[acc_idx] + acc_vals[offset + 30];
    }
  }

  __syncthreads();

  // Have 29*3 values to store
  // They are in acc_vals[0..28,64*30..64*30+28,64*60..64*60+28]
  if (tx < 29) {
    vals[tx] = final_vals[tx];
    vals[29 + tx] = final_vals[29 + tx];
    vals[2 * 29 + tx] = final_vals[2 * 29 + tx];
  }
}


__global__ void BuildDescriptor(float *_valsim, unsigned char *_desc) {
  int p = blockIdx.x;

  size_t idx = threadIdx.x;

  if (idx < 61) {
    float *valsim = &_valsim[3 * 29 * p];

    unsigned char *desc = &_desc[61 * p];

    unsigned char desc_r = 0;

#pragma unroll
    for (int i = 0; i < (idx == 60 ? 6 : 8); ++i) {
      int idx1 = comp_idx_1[idx * 8 + i];
      int idx2 = comp_idx_2[idx * 8 + i];
      desc_r |= (valsim[idx1] > valsim[idx2] ? 1 : 0) << i;
    }

    desc[idx] = desc_r;
  }
}


double ExtractDescriptors(cv::KeyPoint *d_pts, std::vector<CudaImage> &h_imgs, CudaImage *d_imgs,
                          unsigned char *desc_d, float* vals_d, int patsize, int numPts) {
  int size2 = patsize;
  int size3 = ceil(2.0f * patsize / 3.0f);
  int size4 = ceil(0.5f * patsize);
  //int numPts;
  //cudaMemcpyFromSymbol(&numPts, d_PointCounter, sizeof(int));

  // TimerGPU timer0(0);
  dim3 blocks(numPts);
  dim3 threads(EXTRACT_S);

  ExtractDescriptors << <blocks, threads>>>(d_pts, d_imgs, vals_d, size2, size3, size4);
  CHK;

  cudaMemsetAsync(desc_d, 0, numPts * 61);
  BuildDescriptor << <blocks, 64>>> (vals_d, desc_d);
  CHK;

  ////checkMsg("ExtractDescriptors() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("ExtractDescriptors time =     %.2f ms\n", gpuTime);
#endif

  return gpuTime;
}




#define NTHREADS_MATCH 32
__global__ void MatchDescriptors(unsigned char *d1, unsigned char *d2,
                                 int pitch, int nkpts_2, cv::DMatch *matches) {
  int p = blockIdx.x;

  int x = threadIdx.x;

  __shared__ int idxBest[NTHREADS_MATCH];
  __shared__ int idxSecondBest[NTHREADS_MATCH];
  __shared__ int scoreBest[NTHREADS_MATCH];
  __shared__ int scoreSecondBest[NTHREADS_MATCH];

  idxBest[x] = 0;
  idxSecondBest[x] = 0;
  scoreBest[x] = 512;
  scoreSecondBest[x] = 512;

  __syncthreads();

  // curent version fixed with popc, still not convinced
  unsigned long long *d1i = (unsigned long long *)(d1 + pitch * p);

  for (int i = 0; i < nkpts_2; i += NTHREADS_MATCH) {
    unsigned long long *d2i = (unsigned long long *)(d2 + pitch * (x + i));
    if (i + x < nkpts_2) {
      // Check d1[p] with d2[i]
      int score = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        score += __popcll(d1i[j] ^ d2i[j]);
      }
      if (score < scoreBest[x]) {
        scoreSecondBest[x] = scoreBest[x];
        scoreBest[x] = score;
        idxSecondBest[x] = idxBest[x];
        idxBest[x] = i + x;
      } else if (score < scoreSecondBest[x]) {
        scoreSecondBest[x] = score;
        idxSecondBest[x] = i + x;
      }
    }
  }

  //    for( int i=16; i>=1; i/=2) {
  //        int tBest = __shfl_down(scoreBest,i);
  //        int tIdx = __shfl_down(idxBest,i);
  //        if(tBest < scoreBest) {
  //            scoreSecondBest = scoreBest;
  //            idxSecondBest = idxBest;
  //            scoreBest = tBest;
  //            idxBest = tIdx;
  //        }
  //        tBest = __shfl_down(scoreSecondBest,i);
  //        tIdx = __shfl_down(idxSecondBest,i);
  //        if(tBest < scoreSecondBest) {
  //            scoreSecondBest = tBest;
  //            idxSecondBest = tIdx;
  //        }
  //    }

  __syncthreads();

  for (int i = NTHREADS_MATCH / 2; i >= 1; i /= 2) {
    if (x < i) {
      if (scoreBest[x + i] < scoreBest[x]) {
        scoreSecondBest[x] = scoreBest[x];
        scoreBest[x] = scoreBest[x + i];
        idxSecondBest[x] = idxBest[x];
        idxBest[x] = idxBest[x + i];
      } else if (scoreBest[x + i] < scoreSecondBest[x]) {
        scoreSecondBest[x] = scoreBest[x + i];
        idxSecondBest[x] = idxBest[x + i];
      }
      if (scoreSecondBest[x + i] < scoreSecondBest[x]) {
        scoreSecondBest[x] = scoreSecondBest[x + i];
        idxSecondBest[x] = idxSecondBest[x + i];
      }
    }
  }
  // if(i>16) __syncthreads();
  //        if(x<i) {
  //            if( scoreBest[x+i] < scoreSecondBest[x] ) {
  //                scoreSecondBest[x] = scoreBest[x+i];
  //                idxSecondBest[x] = idxBest[x+i];
  //            } else if (scoreSecondBest[x+i] < scoreSecondBest[x] ) {
  //                scoreSecondBest[x] = scoreSecondBest[x+i];
  //                idxSecondBest[x] = idxSecondBest[x+i];
  //            }
  //        }
  //        if(i>16) __syncthreads();
  //}

  /*for (int i = 1; i <= NTHREADS_MATCH; ++i) {
    if (scoreBest[i] < scoreBest[0]) {
      scoreSecondBest[0] = scoreBest[0];
      scoreBest[0] = scoreBest[i];
      idxSecondBest[0] = idxBest[0];
      idxBest[0] = idxBest[i];
    }  else if( scoreBest[i] < scoreSecondBest[0] ) {
         scoreSecondBest[0] = scoreBest[i];
         idxSecondBest[0] = idxBest[i];
     }
     if(scoreSecondBest[i] < scoreSecondBest[0]) {
         scoreSecondBest[0] = scoreSecondBest[i];
         idxSecondBest[0] = idxSecondBest[i];
     }
  }*/

  //    if(x==0) {
  //        matches[2*p].queryIdx = p;
  //        matches[2*p].trainIdx = idxBest;
  //        matches[2*p].distance = scoreBest;
  //        matches[2*p+1].queryIdx = p;
  //        matches[2*p+1].trainIdx = idxSecondBest;
  //        matches[2*p+1].distance = scoreSecondBest;
  //    }

  if (x == 0) {
    matches[2 * p].queryIdx = p;
    matches[2 * p].trainIdx = idxBest[x];
    matches[2 * p].distance = scoreBest[x];
    matches[2 * p + 1].queryIdx = p;
    matches[2 * p + 1].trainIdx = idxSecondBest[x];
    matches[2 * p + 1].distance = scoreSecondBest[x];
  }
}


void MatchDescriptors(cv::Mat &desc_query, cv::Mat &desc_train,
		      std::vector<std::vector<cv::DMatch> > &dmatches,
		      size_t pitch, 
		      unsigned char* descq_d, unsigned char* desct_d, cv::DMatch* dmatches_d, cv::DMatch* dmatches_h) {

    dim3 block(desc_query.rows);
    
    MatchDescriptors << <block, NTHREADS_MATCH>>>(descq_d, desct_d, pitch, desc_train.rows, dmatches_d);

    cudaMemcpy(dmatches_h, dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch),
	       cudaMemcpyDeviceToHost);

    for (int i = 0; i < desc_query.rows; ++i) {
	std::vector<cv::DMatch> tdmatch;
	//std::cout << dmatches_h[2*i].trainIdx << " - " << dmatches_h[2*i].queryIdx << std::endl;
	tdmatch.push_back(dmatches_h[2 * i]);
	tdmatch.push_back(dmatches_h[2 * i + 1]);
	dmatches.push_back(tdmatch);
    }
    
}


void MatchDescriptors(cv::Mat &desc_query, cv::Mat &desc_train,
                      std::vector<std::vector<cv::DMatch> > &dmatches) {
  size_t pitch1, pitch2;
  unsigned char *descq_d;
  cudaMallocPitch(&descq_d, &pitch1, 64, desc_query.rows);
  cudaMemset2D(descq_d, pitch1, 0, 64, desc_query.rows);
  cudaMemcpy2D(descq_d, pitch1, desc_query.data, desc_query.cols,
               desc_query.cols, desc_query.rows, cudaMemcpyHostToDevice);
  unsigned char *desct_d;
  cudaMallocPitch(&desct_d, &pitch2, 64, desc_train.rows);
  cudaMemset2D(desct_d, pitch2, 0, 64, desc_train.rows);
  cudaMemcpy2D(desct_d, pitch2, desc_train.data, desc_train.cols,
               desc_train.cols, desc_train.rows, cudaMemcpyHostToDevice);

  dim3 block(desc_query.rows);

  cv::DMatch *dmatches_d;
  cudaMalloc(&dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch));

  MatchDescriptors << <block, NTHREADS_MATCH>>>(descq_d, desct_d, pitch1, desc_train.rows, dmatches_d);

  cv::DMatch *dmatches_h = new cv::DMatch[2 * desc_query.rows];
  cudaMemcpy(dmatches_h, dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < desc_query.rows; ++i) {
    std::vector<cv::DMatch> tdmatch;
    //std::cout << dmatches_h[2*i].trainIdx << " - " << dmatches_h[2*i].queryIdx << std::endl;
    tdmatch.push_back(dmatches_h[2 * i]);
    tdmatch.push_back(dmatches_h[2 * i + 1]);
    dmatches.push_back(tdmatch);
  }

  cudaFree(descq_d);
  cudaFree(desct_d);
  cudaFree(dmatches_d);

  delete[] dmatches_h;
}


void InitCompareIndices() {

    int comp_idx_1_h[61 * 8];
    int comp_idx_2_h[61 * 8];

    int cntr = 0;
    for (int j = 0; j < 4; ++j) {
      for (int i = j + 1; i < 4; ++i) {
        comp_idx_1_h[cntr] = 3 * j;
        comp_idx_2_h[cntr] = 3 * i;
        cntr++;
      }
    }
    for (int j = 0; j < 3; ++j) {
      for (int i = j + 1; i < 4; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 1;
        comp_idx_2_h[cntr] = 3 * i + 1;
        cntr++;
      }
    }
    for (int j = 0; j < 3; ++j) {
      for (int i = j + 1; i < 4; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 2;
        comp_idx_2_h[cntr] = 3 * i + 2;
        cntr++;
      }
    }

    // 3x3
    for (int j = 4; j < 12; ++j) {
      for (int i = j + 1; i < 13; ++i) {
        comp_idx_1_h[cntr] = 3 * j;
        comp_idx_2_h[cntr] = 3 * i;
        cntr++;
      }
    }
    for (int j = 4; j < 12; ++j) {
      for (int i = j + 1; i < 13; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 1;
        comp_idx_2_h[cntr] = 3 * i + 1;
        cntr++;
      }
    }
    for (int j = 4; j < 12; ++j) {
      for (int i = j + 1; i < 13; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 2;
        comp_idx_2_h[cntr] = 3 * i + 2;
        cntr++;
      }
    }

    // 4x4
    for (int j = 13; j < 28; ++j) {
      for (int i = j + 1; i < 29; ++i) {
        comp_idx_1_h[cntr] = 3 * j;
        comp_idx_2_h[cntr] = 3 * i;
        cntr++;
      }
    }
    for (int j = 13; j < 28; ++j) {
      for (int i = j + 1; i < 29; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 1;
        comp_idx_2_h[cntr] = 3 * i + 1;
        cntr++;
      }
    }
    for (int j = 13; j < 28; ++j) {
      for (int i = j + 1; i < 29; ++i) {
        comp_idx_1_h[cntr] = 3 * j + 2;
        comp_idx_2_h[cntr] = 3 * i + 2;
        cntr++;
      }
    }

    cudaMemcpyToSymbol(comp_idx_1, comp_idx_1_h, 8 * 61 * sizeof(int));
    cudaMemcpyToSymbol(comp_idx_2, comp_idx_2_h, 8 * 61 * sizeof(int));

}


__global__ void FindOrientation(cv::KeyPoint *d_pts, CudaImage *d_imgs) {
  __shared__ float resx[42], resy[42];
  __shared__ float re8x[42], re8y[42];
  int p = blockIdx.x;
  int tx = threadIdx.x;
  if (tx < 42) resx[tx] = resy[tx] = 0.0f;
  __syncthreads();
  int lev = d_pts[p].class_id;
  float *dxd = d_imgs[4 * lev + 2].d_data;
  float *dyd = d_imgs[4 * lev + 3].d_data;
  int pitch = d_imgs[4 * lev + 0].pitch;
  int octave = d_pts[p].octave;
  int step = (int)(0.5f * d_pts[p].size + 0.5f) >> octave;
  int x = (int)(d_pts[p].pt.x + 0.5f) >> octave;
  int y = (int)(d_pts[p].pt.y + 0.5f) >> octave;
  int i = (tx & 15) - 6;
  int j = (tx / 16) - 6;
  int r2 = i * i + j * j;
  if (r2 < 36) {
    float gweight = exp(-r2 / (2.5f * 2.5f * 2.0f));
    int pos = (y + step * j) * pitch + (x + step * i);
    float dx = gweight * dxd[pos];
    float dy = gweight * dyd[pos];
    float angle = atan2(dy, dx);
    int a = max(min((int)(angle * (21 / CV_PI)) + 21, 41), 0);
    atomicAdd(resx + a, dx);
    atomicAdd(resy + a, dy);
  }
  __syncthreads();
  if (tx < 42) {
    re8x[tx] = resx[tx];
    re8y[tx] = resy[tx];
    for (int k = tx + 1; k < tx + 7; k++) {
      re8x[tx] += resx[k < 42 ? k : k - 42];
      re8y[tx] += resy[k < 42 ? k : k - 42];
    }
  }
  __syncthreads();
  if (tx == 0) {
    float maxr = 0.0f;
    int maxk = 0;
    for (int k = 0; k < 42; k++) {
      float r = re8x[k] * re8x[k] + re8y[k] * re8y[k];
      if (r > maxr) {
        maxr = r;
        maxk = k;
      }
    }
    float angle = atan2(re8y[maxk], re8x[maxk]);
    d_pts[p].angle = (angle < 0.0f ? angle + 2.0f * CV_PI : angle);
    // printf("XXX %.2f %.2f %.2f\n", d_pts[p].pt.x, d_pts[p].pt.y,
    // d_pts[p].angle/CV_PI*180.0f);
  }
}

double FindOrientation(cv::KeyPoint *d_pts, std::vector<CudaImage> &h_imgs, CudaImage *d_imgs, int numPts) {

  safeCall(cudaMemcpyAsync(d_imgs, (float *)&h_imgs[0],
                           sizeof(CudaImage) * h_imgs.size(),
                           cudaMemcpyHostToDevice));

  // TimerGPU timer0(0);
  cudaStreamSynchronize(0);
  dim3 blocks(numPts);
  dim3 threads(ORIENT_S);
  FindOrientation << <blocks, threads>>> (d_pts, d_imgs);
  CHK
  // checkMsg("FindOrientation() execution failed\n");
  // safeCall(cudaThreadSynchronize());
  double gpuTime = 0;  // timer0.read();
#ifdef VERBOSE
  printf("FindOrientation time =        %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}
