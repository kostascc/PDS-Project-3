/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

#include "GPU.cuh"


using namespace std;


namespace GPU
{


	int run(Parameters params)
	{


#ifdef DEBUG
		cout << "GPU Starting \n";
#endif

		// Start Clock
		utils::Clock clock = utils::Clock();
		clock.startClock();

		utils::ImageFile img = utils::ImageFile();
		img.Read(params.input.imgPath);

#ifdef DEBUG
		cout << "Image: " << img.width << "x" << img.width << " (Pixels: " << pow(img.width, 2) << ")\n";
#endif

#ifdef USE_LOG_FILE
		ofstream log;
		log.open("./livelog_GPU.txt");
#endif

		// TODO: nvcc/linker goes balls with this line for some reason..
		//float sigmaSquared = (float)pow(params.algorithm.sigma, 2);
		float sigmaSquared = params.algorithm.sigma * params.algorithm.sigma;


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
		cudaMalloc(&pixN_d, img.width * img.width * sizeof(float));
		// Copy Pixels from Image File
		cudaMemcpy(pixN_d, img.pixelArr, img.width * img.width * sizeof(float), cudaMemcpyHostToDevice);


		// Device' resulting Pixels
		float* pix_d;
		cudaMalloc(&pix_d, img.width * img.width * sizeof(float));
		//cudaMallocManaged(&pix_d, sizeof(Parameters)); // not managed, so it stays between kernel calls
		// Set Pixels to 0.0f
		cudaMemset(pix_d, 0.0f, img.width * img.width * sizeof(float));



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
		float* wSum_d;
		cudaMalloc(&wSum_d, (img.width - params.algorithm.patchSize) * (img.width - params.algorithm.patchSize) * sizeof(float));
		// Copy Pixels from Image File
		cudaMemset(wSum_d, 0.0f, (img.width - params.algorithm.patchSize) * (img.width - params.algorithm.patchSize) * sizeof(float));


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
		int sharedBytes = (1 + pow(params.algorithm.patchSize, 2)) * sizeof(float);


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
		dim3 blocks(img.width, img.height);

#ifdef DEBUG
		cout << "Weight Sum Kernel:\n"
			<< "> Blocks / Threads / Patches: "
			<< "[" << blocks.x << "," << blocks.y << "] / "
			<< "[" << threads.x << "," << threads.y << "] / "
			<< pow(img.width - params.algorithm.patchSize, 2) << "\n"
			<< "> Shared Memory Per ThreadBlock: " << sharedBytes << " B\n";
#endif



		kernelWeightSum __KERNEL3(blocks, threads, sharedBytes) (wSum_d, pixN_d, img.width, params.algorithm.patchSize, sigmaSquared);
		cudaDeviceSynchronize();

		float* wSum = (float*)malloc((img.width - params.algorithm.patchSize) * (img.width - params.algorithm.patchSize) * sizeof(float));
		cudaMemcpy(wSum, wSum_d, (img.width - params.algorithm.patchSize) * (img.width - params.algorithm.patchSize) * sizeof(float), cudaMemcpyDeviceToHost);

		for (int i = 0; i < img.width - params.algorithm.patchSize; i++)
		{
			for (int j = 0; j < img.width - params.algorithm.patchSize; j++)
			{
				log << wSum[i * (img.width - params.algorithm.patchSize) + j] << " ";
			}
			log << "\n";
		}

		log << "\n\n\n";

		// Copy Resulting Pixels into hosts' image matrix
		cudaMemcpy(img.pixelArr, pix_d, img.width * img.width * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(pixN_d);
		cudaFree(wSum_d);
		cudaFree(pix_d);

		// Save Image
		img.Write(params.input.outputDir + "/sigma" + to_string(params.algorithm.sigma) + "_" + utils::ImageFile::GetFileName(params.input.imgPath));


		cout << "GPU Took " << clock.stopClock() << "\n";



#ifdef USE_LOG_FILE

		log << "\n\n";

		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				log << pix[i * 16 + j] << " ";
			}
			log << "\n";
		}

		log.close();
#endif

		return 0;
	}



	__global__ void kernelWeightSum(float* wSum_d, float* pixN_d, int imgWidth, int patchSize, float sigmaSquared)
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
		if (!threadIdx.x && !threadIdx.y)
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


		//int idx = blockIdx.x + threadIdx.x;	// Horizontal Coordinate of Patch
		//int idy = blockIdx.y + threadIdx.y; // vertical Coordinate of Patch

		// Noisy Pixel Patch
		float* pixF = (float*)&_shmem[1];
		if (threadIdx.x < patchSize && threadIdx.y < patchSize)
			pixF[threadIdx.x * patchSize + threadIdx.y] =
			pixN_d[(blockIdx.x + threadIdx.x) * imgWidth + (blockIdx.y + threadIdx.y)];
		__syncthreads();


		// Patch Coordinates (Upper Left Corner)
		int x, y;


		for (int j = 0; j < THREADS_Y; j++)
			for (int i = 0; i < THREADS_X; i++)
			{
				x = (threadIdx.x * THREADS_X + i);
				y = (threadIdx.y * THREADS_Y + j);

				// Sum Only patches of different coordinates and 
				// within boundaries
				// (don't weight the patch with itself)
				if (blockIdx.x != x && blockIdx.y != y
					&& x < imgWidth && y < imgWidth)
				{
					// Patch at coordinates (x,y)

					float d = 0.0f;

					// For Each pixel in a patch
					for (int yy = 0; yy < patchSize; yy++)
						for (int xx = 0; xx < patchSize; xx++)
						{
							int a = pixF[yy * patchSize + xx];
							int b = pixN_d[(y + yy) * imgWidth + (x + xx)];
							a = a - b;
							d += a * a;
						}

					// Get Exponential
					d = exp(-d / sigmaSquared);
					atomicAdd(wSum, d);

				}
			}
		__syncthreads();

		// Save wSum, based on the pixel coordinates
		if(!threadIdx.x && !threadIdx.y)
		wSum_d[blockIdx.y * (imgWidth - patchSize) + blockIdx.x] += wSum[0];

		//__syncthreads();

		return;
	}



	__global__ void kernelPatchPixels(float* wSum_d, float* pixN_d, int imgWidth, int patchSize, float sigmaSquared)
	{

		// Shared Memory
		extern __shared__ float _shmem[];


		return;
	}


	int iDivUp(int a, int b)
	{
		return ((a % b) != 0) ? (a / b + 1) : (a / b);
	}




}


