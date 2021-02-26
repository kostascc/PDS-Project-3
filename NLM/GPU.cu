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

		return;
	}


	int iDivUp(int a, int b)
	{
		return ((a % b) != 0) ? (a / b + 1) : (a / b);
	}




}


