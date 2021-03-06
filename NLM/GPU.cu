/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

#include "GPU.cuh"


using namespace std;


namespace GPU
{

	/**
	* Error Checking for Kernels,
	* provided by nvidia in CUDA samples.
	**/
#ifndef cudaCheckErrors
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)
#endif


	/**
	* Inline Error Checking for Kernels,
	* provided by and widely available in forums.
	**/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) exit(code);
		}
	}




	int run(Parameters params)
	{


#ifdef DEBUG
		cout << "GPU Starting \n";
		cout << "Patch Size: " << PATCH_SIZE << "x" << PATCH_SIZE << "\n";
#endif

		// Fetch Image
		utils::ImageFile img = utils::ImageFile();
		img.Read(params.input.imgPath);


#ifdef DEBUG
		cout << "Image: " << img.width << "x" << img.width << " (Pixels: " << POW2(img.width) << ")\n";
#endif

#ifdef USE_LOG_FILE
		ofstream log;
		log.open("./livelog_GPU.txt");
#endif

		// TODO: nvcc/linker can't comprehend this line, why?
		//float sigmaSquared = (float)pow(params.algorithm.sigma, 2);


		/*********************************************
		 *   Pixels (Noisy / Resulting)
		 *
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~  ← [img_height]
		 *                 ↑
		 *              [img_width]
		 *********************************************/

		 // Device' Noisy Pixels
		float* pixN_d;
		cudaMalloc(&pixN_d, POW2(img.width) * sizeof(float));
		cudaCheckErrors("malloc: pixN_d");
		// Copy Pixels from Image File
		cudaMemcpy(pixN_d, img.pixelArr, POW2(img.width) * sizeof(float), cudaMemcpyHostToDevice);
		cudaCheckErrors("Memcpy: pixN_d");

		// Device' resulting Pixels
		float* pix_d;
		cudaMalloc(&pix_d, POW2(img.width) * sizeof(float));
		cudaCheckErrors("Malloc: pix_d");
		// Set Pixels to 0.0f
		cudaMemset(pix_d, 0.0f, POW2(img.width) * sizeof(float));
		cudaCheckErrors("Memset: pix_d");


		/*************************************************
		 *   Shared Memory (per ThreadBlock)
		 *
		 *    +---+---+---+---+---+-------+---+---+---+---+---+---+-------+---+---+
		 *    | o | o | o | o | o | . . . | o | o | o | o | o | o | . . . | o | o |
		 *    +---+---+---+---+---+-------+---+---+---+---+---+---+-------+---+---+
		 *      ⥎  ⬑---------------------------⬏  ⬑---------------------------⬏
		 *     wSum       pixF (2D patch)                  pixR (2D patch)
		 *
		 *   [ 1 + 2 * (patch_size * patch_size) ] * <float>
		 *************************************************/
		 // Shared memory (Bytes)
		int sharedBytes = (1 + 2 * POW2(PATCH_SIZE)) * sizeof(float);


		/*************************************************
		 *   Blocks
		 *   +---+---+---+---+
		 *   | B |   |   |   |
		 *   +---------------+
		 *   |   |   |   |   |
		 *   +---------------+
		 *   |   |   |   |   |
		 *   +---------------+
		 *   |   |   |   |   |
		 *   +---+---+---+---+  ← [img_height]
		 *                   ↑
		 *              [img_width]
		 *
		 *
		 *   Threads (for each block)
		 *   +---+---+---+
		 *   | T |   |   |
		 *   +-----------+
		 *   |   |   |   |
		 *   +-----------+
		 *   |   |   |   |
		 *   +-----------+  ← [THREADS_Y]
		 *               ↑
		 *          [THREADS_X]
		 *
		 *
		 *
		 *  Patches (for each Thread)
		 *
		 *   +---+---+---+---+
		 *   | a | a | b | b |
		 *   +---------------+
		 *   | a | a | b | b |
		 *   +-----------+---+
		 *   | c | c |       |
		 *   +-------+  ...  |
		 *   | c | c |       |
		 *   +---+---+-------+  ← [img_height / THREADS_Y]
		 *                   ↑
		 *          [img_width / THREADS_X]
		 *
		 *   All 'a' patches will be executed by the same thread.
		 *   The same for 'b' patches etc.
		 *
		 *
		 *************************************************/
		dim3 threads(THREADS_X, THREADS_Y);
		dim3 blocks(img.width - PATCH_SIZE + 1, img.height - PATCH_SIZE + 1);

		// Display Kernel Info
#ifdef DEBUG
		cout << "Weight Sum Kernel:\n"
			<< "> Blocks / Threads / Patches: "
			<< "[" << blocks.x << "," << blocks.y << "] / "
			<< "[" << threads.x << "," << threads.y << "] / "
			<< POW2(img.width - PATCH_SIZE) << "\n"
			<< "> Shared Memory Per ThreadBlock: " << sharedBytes << " B\n"
			<< "> Total Shared Memory: " << blocks.x * blocks.y * sharedBytes / (1 << 10) << " KB\n";
#endif

		// Start Clock
		utils::Clock clock = utils::Clock();
		clock.startClock();

		// __KERNEL3(a,b,c)  is translated to  <<<a,b,c>>>
		// it has been defined as such, because intellisense doesn't recognize CUDA keywords
		// See GPU.cuh definitions

		// Run Kernel
		kernel __KERNEL3(blocks, threads, sharedBytes) (pix_d, pixN_d, img.width, POW2(params.algorithm.sigma));
		cudaCheckErrors("Kernel: 1 Initialize");
		cudaDeviceSynchronize();
		cudaCheckErrors("Kernel: 1 Synchronize");

		// Print Timer
		cout << "GPU Took " << clock.stopClock() << "\n";

		// Copy Resulting Pixels into hosts' image matrix
		cudaMemcpy(img.pixelArr, pix_d, POW2(img.width) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaCheckErrors("Memcpy: pix_d (To Host)");

		// Free CUDA Global Memory
		cudaFree(pixN_d);
		cudaCheckErrors("Free: pixN_d");
		cudaFree(pix_d);
		cudaCheckErrors("Free: pix_d");

		// Save Image
		img.Write(params.input.outputDir + "/GPU_sigma" + to_string(params.algorithm.sigma) +
			"_patch" + to_string(PATCH_SIZE) + "__" + utils::ImageFile::GetFileName(params.input.imgPath));

		return 0;
	}



	__global__ void kernel(float* pix_d, float* pixN_d, int imgWidth, float sigmaSquared)
	{

		// Shared Memory
		extern __shared__ float _shmem[];

		/*******************************************************************
		 *      wSum                              _shmem
		 * +------------+          +---+---+---+---+---+-------+---+---+
		 * | &_shmem[0] |   +->    | o |   |   |   |   | . . . |   |   |
		 * +------------+          +---+---+---+---+---+-------+---+---+
		 *                           ↑
		 *                       _shmem[0]
		 *******************************************************************/

		 // Final Sum of Weights (whithin the block)
		float* wSum = &_shmem[0];
		if (!threadIdx.y && !threadIdx.x)
			*wSum = 0.0f;


		/*******************************************************************
		 *      pixF                              _shmem
		 * +------------+          +---+---+---+---+---+-------+---+---+---+-------+---+---+
		 * | &_shmem[1] |   +->    |   | o | o | o | o | . . . | o | o |   | . . . |   |   |
		 * +------------+          +---+---+---+---+---+-------+---+---+---+-------+---+---+
		 *                               ↑                           ↑
		 *                           _shmem[1]   ...   _shmem[1+patch_size^2-1]
		 *******************************************************************/

		 // Noisy Pixel Patch
		float* pixF = (float*)&_shmem[1];
		if (threadIdx.y < PATCH_SIZE && threadIdx.x < PATCH_SIZE)

			pixF[threadIdx.y * PATCH_SIZE + threadIdx.x]
			= pixN_d[(blockIdx.y + threadIdx.y) * imgWidth + (blockIdx.x + threadIdx.x)];

		__syncthreads();


		/*******************************************************************
		 *      pixF                              _shmem
		 * +------------+          +---+---+-------+---+---+---+---+---+---+-------+---+---+
		 * | &_shmem[1] |   +->    |   |   | . . . |   |   | o | o | o | o | . . . | o | o |
		 * +------------+          +---+---+-------+---+---+---+---+---+---+-------+---+---+
		 *                                                   ↑                           ↑
		 *                                  _shmem[1+patch_size^2]   ...   _shmem[1+2*patch_size^2-1]
		 *******************************************************************/

		float* pixR = (float*)&_shmem[1 + POW2(PATCH_SIZE)];
		if (threadIdx.y < PATCH_SIZE && threadIdx.x < PATCH_SIZE)

			pixR[threadIdx.y * PATCH_SIZE + threadIdx.x] = 0.0f;

		__syncthreads();


		// Globally Unique ID (not used)
		//int uidx, uidy;
		//uidx = blockIdx.x * blockDim.x + threadIdx.x;
		//uidy = blockIdx.y * blockDim.y + threadIdx.y;

		// A patch Window Consists of 
		// [imgWidth / patchSize / THREADS_X , imgWidth / patchSize / THREADS_Y]
		// patches.

		// Pixel Coordinates: (blockIdx.y, blockIdx.x)


		// Number of patches to be cheked in a thread
		int patchesY = ((imgWidth - PATCH_SIZE + THREADS_Y)) / THREADS_Y;	// Number of Patches Verticaly
		int patchesX = ((imgWidth - PATCH_SIZE + THREADS_Y)) / THREADS_X;	// Number of Patches Horizontaly

		// For Each one of those threads
		for (int j = 0; j < patchesY; j++)
			for (int i = 0; i < patchesX; i++)
			{

				int px = threadIdx.x * patchesY + i;	// Patch Coordinates
				int py = threadIdx.y * patchesY + j;	// Patch Coordinates

				__syncthreads();


				// Sum Only patches of different coordinates and 
				// within boundaries
				// (don't weight the patch with itself)
				if (px < imgWidth - PATCH_SIZE && py < imgWidth - PATCH_SIZE
					&& (blockIdx.x != px || blockIdx.y != py))
				{
					// Patch at coordinates (px,py)
					float d = 0.0f;

					// For Each pixel in the patch
#pragma unroll
					for (int yy = 0; yy < PATCH_SIZE; yy++)
					{
#pragma unroll
						for (int xx = 0; xx < PATCH_SIZE; xx++)
						{
							// Substract vectors 
							float a = __fsub_rn(
								pixF[yy * PATCH_SIZE + xx],
								pixN_d[(py + yy) * imgWidth + (px + xx)]
							);
							// Find Power of 2, add to squared Distance
							d += POW2(a);
						}
					}

					// Exponential Distance
					d = __expf(-d / sigmaSquared);
					atomicAdd(wSum, d);


					// For Each pixel in the same patch
#pragma unroll
					for (int yy = 0; yy < PATCH_SIZE; yy++)
					{
#pragma unroll
						for (int xx = 0; xx < PATCH_SIZE; xx++)
						{
							// Apply weighted pixel to shared memory
							atomicAdd(&pixR[yy * PATCH_SIZE + xx], d * pixN_d[(py + yy) * imgWidth + (px + xx)]);
						}
					}

				}
			}

		// Synchronize threads to get
		// the correct weight sum
		__syncthreads();


		// Copy pixels from Shared Memory to resulting image
		if (threadIdx.x < PATCH_SIZE && threadIdx.y < PATCH_SIZE && blockIdx.x < imgWidth && blockIdx.y < imgWidth)
			pix_d[(blockIdx.y + threadIdx.y) * imgWidth + blockIdx.x + threadIdx.x] = pixR[threadIdx.y * PATCH_SIZE + threadIdx.x] / *wSum;

		return;
	}



	int iDivUp(int a, int b)
	{
		return ((a % b) != 0) ? (a / b + 1) : (a / b);
	}




}


