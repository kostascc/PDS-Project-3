/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

#include "GPU.cuh"


using namespace std;


namespace GPU
{

	// Error Checking
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



	int run(Parameters params)
	{


#ifdef DEBUG
		cout << "GPU Starting \n";
#endif


		utils::ImageFile img = utils::ImageFile();
		img.Read(params.input.imgPath);

#ifdef DEBUG
		cout << "Image: " << img.width << "x" << img.width << " (Pixels: " << POW2(img.width) << ")\n";
#endif

#ifdef USE_LOG_FILE
		ofstream log;
		log.open("./livelog_GPU.txt");
#endif

		// Start Clock
		utils::Clock clock = utils::Clock();
		clock.startClock();

		// TODO: nvcc/linker goes balls with this line for some reason..
		//float sigmaSquared = (float)pow(params.algorithm.sigma, 2);
		float sigmaSquared = POW2(params.algorithm.sigma);


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



		/*********************************************
		 *   Weight Sums
		 *
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~
		 *   ~ ~ ~ ~ ~ ~ ~ ~  ← [img_height - patch_size]
		 *                 ↑
		 *         [img_width - patch_size]
		 *********************************************/


		 // Device's Weight Sums
		 // (img_width-patch_size) * (img_width-patch_size)
		//float* wSum_d;
		//cudaMalloc(&wSum_d, (img.width - PATCH_SIZE) * (img.width - PATCH_SIZE) * sizeof(float));
		//cudaCheckErrors("Malloc: wSum_d");
		//// Set Weight 
		//cudaMemset(wSum_d, 0.0f, (img.width - PATCH_SIZE) * (img.width - PATCH_SIZE) * sizeof(float));
		//cudaCheckErrors("Memset: wSum_d");

		/*************************************************
		 *   Shared Memory (per ThreadBlock)
		 *
		 *    +---+---+---+---+---+-------+---+---+
		 *    | o | o | o | o | o | . . . | o | o |
		 *    +---+---+---+---+---+-------+---+---+
		 *      ⥎   ⬑--------------------------⬏
		 *     wSum        pixN_d (2D patch)
		 *
		 *   [ 1 + (patch_size * patch_size) ] * <float>
		 *************************************************/
		 // Shared memory (Bytes)
		int sharedBytes = (1 + POW2(PATCH_SIZE)) * sizeof(float);


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
		dim3 blocks(img.width - PATCH_SIZE, img.height - PATCH_SIZE);

		// Display Kernel Info
#ifdef DEBUG
		cout << "Weight Sum Kernel:\n"
			<< "> Blocks / Threads / Patches: "
			<< "[" << blocks.x << "," << blocks.y << "] / "
			<< "[" << threads.x << "," << threads.y << "] / "
			<< POW2(img.width - PATCH_SIZE) << "\n"
			<< "> Shared Memory Per ThreadBlock: " << sharedBytes << " B\n";
#endif


		// Run Kernel
		kernelWeightSum __KERNEL3(blocks, threads, sharedBytesWSum) (pix_d, pixN_d, img.width, PATCH_SIZE, sigmaSquared);
		cudaDeviceSynchronize();
		cudaCheckErrors("Kernel: WeightSum");


		// Copy Resulting Pixels into hosts' image matrix
		cudaMemcpy(img.pixelArr, pix_d, POW2(img.width) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaCheckErrors("Memcpy: pix_d (To Host)");

		cudaFree(pixN_d);
		cudaCheckErrors("Free: pixN_d");
		cudaFree(pix_d);
		cudaCheckErrors("Free: pix_d");


		cout << "GPU Took " << clock.stopClock() << "\n";

		// Save Image
		img.Write(params.input.outputDir + "/GPU_sigma" + to_string(params.algorithm.sigma) + "_" + utils::ImageFile::GetFileName(params.input.imgPath));


		return 0;
	}



	__global__ void kernelWeightSum(float* pix_d, float* pixN_d, int imgWidth, int patchSize, float sigmaSquared)
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

		__syncthreads();

		/*******************************************************************
		 *      pixF                              _shmem
		 * +------------+          +---+---+---+---+---+-------+---+---+
		 * | &_shmem[1] |   +->    |   | o | o | o | o | . . . | o | o |
		 * +------------+          +---+---+---+---+---+-------+---+---+
		 *                               ↑                           ↑
		 *                           _shmem[1]   ...   _shmem[1+patch_size^2-1]
		 *******************************************************************/

		 // Patch Window Coordinates (Left Uppoer Corner)
		/*int idx, idy;
		idx = blockIdx.x * blockDim.x + threadIdx.x;
		idy = blockIdx.y * blockDim.y + threadIdx.y;*/

		// A patch Window Consists of 
		// [imgWidth / patchSize / THREADS_X , imgWidth / patchSize / THREADS_Y]
		// patches.

		// Pixel Coordinates: (blockIdx.y, blockIdx.x)

		// Noisy Pixel Patch
		float* pixF = (float*)&_shmem[1];
		if (threadIdx.y < PATCH_SIZE && threadIdx.x < PATCH_SIZE)

			pixF[threadIdx.y * PATCH_SIZE + threadIdx.x]
			= pixN_d[(blockIdx.y + threadIdx.y) * imgWidth + (blockIdx.x + threadIdx.x)];


		// Number of patches to be cheked in a thread
		int patchesY = ((imgWidth - PATCH_SIZE + THREADS_Y)) / THREADS_Y;	// Number of Patches Verticaly
		int patchesX = ((imgWidth - PATCH_SIZE + THREADS_Y)) / THREADS_X;	// Number of Patches Horizontaly


		// Distant-Patch Coordinates (Upper Left Corner)
		int px, py;


		__syncthreads();

		// For Each Patch
		for (int j = 0; j < patchesY; j++)
			for (int i = 0; i < patchesX; i++)
			{
				px = threadIdx.x * patchesY + i;
				py = threadIdx.y * patchesY + j;

				// Sum Only patches of different coordinates and 
				// within boundaries
				// (don't weight the patch with itself)
				if ((blockIdx.x != px || blockIdx.y != py)
					&& px < imgWidth - PATCH_SIZE && py < imgWidth - PATCH_SIZE)
				{
					// Patch at coordinates (px,py)

					float d = 0.0f;	// Squuared Distance between pixF and the patch


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
							d += __fmul_rn(a, a);
						}
					}

					// Exponential Distance
					d = __expf(-d / sigmaSquared);
					//w[j * patchesY + j] = d;
					atomicAdd(wSum, d);

				}
			}

		// Synchronize threads to get
		// the correct weight sum
		__syncthreads();


		// For Each Patch
		for (int j = 0; j < patchesY; j++)
			for (int i = 0; i < patchesX; i++)
			{
				px = threadIdx.x * patchesY + i;
				py = threadIdx.y * patchesY + j;


				// Sum Only patches of different coordinates and 
				// within boundaries
				// (don't weight the patch with itself)
				if ((blockIdx.x != px || blockIdx.y != py)
					&& px < imgWidth - PATCH_SIZE && py < imgWidth - PATCH_SIZE)
				{
					// Patch at coordinates (px,py)

					float d = 0.0f;	// Squuared Distance between pixF and the patch


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
							d += __fmul_rn(a, a);
						}
					}

					// Exponential Distance
					d = __expf(-d / sigmaSquared);
					d = __fdiv_rn(d, *wSum);

					// For Each pixel in the same patch
					for (int yy = 0; yy < PATCH_SIZE; yy++)
						for (int xx = 0; xx < PATCH_SIZE; xx++)
						{
							// Apply weighted pixel to shared memory
							pixF[yy * PATCH_SIZE + xx] += __fmul_rn(d, pixN_d[(py + yy) * imgWidth + (px + xx)]);
						}

				}
			}

		__syncthreads();

		// Copy pixels from Shared Memory to resulting image
		if (threadIdx.x < PATCH_SIZE && threadIdx.y < PATCH_SIZE)
			pix_d[(blockIdx.y + threadIdx.y) * imgWidth + blockIdx.x + threadIdx.x] = pixF[threadIdx.y * PATCH_SIZE + threadIdx.x];


		return;
	}



	int iDivUp(int a, int b)
	{
		return ((a % b) != 0) ? (a / b + 1) : (a / b);
	}




}


