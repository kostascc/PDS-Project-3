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
#include "vector_functions.h"


//#define DEBUG
//#define USE_LOG_FILE


// Block Dimensions: 8x8 threads
//   = 64 threads
//   = 2 warps
#define THREADS_X 8
#define THREADS_Y 8


/**
 * Workaround for hiding CUDA function and keyword errors, in 
 * Intellisense environments, such as Visual Studio
 **/
#ifdef __INTELLISENSE__
// Intellisense
#define __KERNEL2(grid, block)
#define __KERNEL3(grid, block, sh_mem)
#define __KERNEL4(grid, block, sh_mem, stream)
#define __syncthreads() ;
#define atomicAdd(a,b) *a+=b
#define __expf(a) exp(a)
#define __fsub_rn(a,b) a-b
#define __fadd_rn(a,b) a+b
#define __fmul_rn(a,b) a*b
#define __fdiv_rn(a,b) a/b
#else
// CUDA
#define __KERNEL2(grid, block) <<< grid, block >>>
#define __KERNEL3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define __KERNEL4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif


using namespace std;
using namespace parameters;

namespace GPU
{

	int run(Parameters params);

	/**
	 * Cuda Kernel
	 * @param pix_d Pixels of Resulting Image
	 * @param pixN_d Pixels of Noisy Image
	 * @param imgWidth Width and Height of Image (Pixels)
	 * @param sigmaSquared The sigma parameter squareed
	 **/
	__global__ void kernel(float* pix_d, float* pixN_d, int imgWidth, float sigmaSquared);

	/**
	 * Divide with upwards rounding
	 * Provided by nvidia in CUDA samples
	 **/
	int iDivUp(int a, int b);

}


#endif
