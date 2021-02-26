/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/
 
#ifndef GPU_CUH
#define GPU_CUH

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "vector_functions.h"
#include "Utils.h"
#include "Parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <math.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"
//#include "cuda/atomic"
//#include "cuda/std/atomic"




#define DEBUG
#define USE_LOG_FILE


// Block Dimensions: 8x8 threads
//   = 64 threads
//   = 2 warps
#define THREADS_X 8
#define THREADS_Y 8


// Workaround for Intellisense errors
#ifdef __INTELLISENSE__
#define __KERNEL2(grid, block)
#define __KERNEL3(grid, block, sh_mem)
#define __KERNEL4(grid, block, sh_mem, stream)
#define __syncthreads();
#define atomicAdd(a,b);
#else
#define __KERNEL2(grid, block) <<< grid, block >>>
#define __KERNEL3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define __KERNEL4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif


using namespace std;
using namespace parameters;

namespace GPU
{

	int run(Parameters params);

	__global__ void kernelWeightSum(float* wSum_d, float* pixN_d, int imgWidth, int patchSize, float sigmaSquared);

	__global__ void kernelPatchPixels(float* pix_d, float* wSum_d, float* pixN_d, int imgWidth, int patchSize, float sigmaSquared);

	int iDivUp(int a, int b);

}


#endif
