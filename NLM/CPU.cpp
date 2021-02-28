/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

#include "CPU.h"

using namespace std;


namespace CPU
{

	int run(Parameters params)
	{

#ifdef DEBUG
		cout << "CPU Starting \n";
#endif

		utils::ImageFile img = utils::ImageFile();
		img.Read(params.input.imgPath);


#ifdef USE_LOG_FILE
		ofstream log;
		log.open("./livelog_CPU.txt");
#endif


		int patchSizeFloat = POW2(PATCH_SIZE) * sizeof(float);

		float* pix = (float*)calloc(POW2(img.width), sizeof(float));

		float sigmaSquared = POW2(params.algorithm.sigma);

		// Width of weight matrix
		int wMapWidth = img.width - PATCH_SIZE;

		// Start Clock
		utils::Clock clock = utils::Clock();
		clock.startClock();

		// For Each patch of size 'patchSize'

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
		for (int y_f = 0; y_f < img.width - PATCH_SIZE + 1; y_f++)
		{
			cout << ".";
			for (int x_f = 0; x_f < img.width - PATCH_SIZE + 1; x_f++)
			{
				// x_f, y_f: Coordinates of patch starting pixel
				// x: Width
				// y: height



				// Create Pixel Matrix of Patch
				float* pixF = (float*)malloc(patchSizeFloat);
				for (int i = 0; i < PATCH_SIZE; i++)
					memcpy(
						&pixF[i * PATCH_SIZE],
						&img.pixelArr[(i + y_f) * img.width + x_f],
						PATCH_SIZE * sizeof(float));


				// Sum of weights
				double wSum = 0.0f;


				// Create Weight map
				double* w = (double*)calloc(POW2(wMapWidth), sizeof(double));


				// for Each other patch
				for (int y_g = 0; y_g < wMapWidth; y_g++)
				{
					for (int x_g = 0; x_g < wMapWidth; x_g++)
					{
						// x_w, y_w: Coordinates of patch starting pixel
						// x: Width
						// y: height

						// Dont Compare the same window
						if (x_g == x_f && y_g == y_f)
						{
							w[y_g * wMapWidth + x_g] = 0.0f;
							continue;
						}


						// Create Pixel Array of new patch
						float* pixG = (float*)malloc(patchSizeFloat);
						for (int i = 0; i < PATCH_SIZE; i++)
							memcpy(
								&pixG[i * PATCH_SIZE],
								&img.pixelArr[(i + y_g) * img.width + x_g],
								PATCH_SIZE * sizeof(float));


						// Find Euclidean Distance squared
						double d = 0.0f;
						for (int i = 0; i < POW2(PATCH_SIZE); i++)
						{

							d += POW2((pixF[i] - pixG[i]));
						}

						d = expf(-d / (sigmaSquared));

						// Calculate non-normalized weight
						w[y_g * wMapWidth + x_g] = d;
						wSum += d;


						free(pixG);
						pixG = NULL;

					}
				}



				// Clean pixels in batch
				for (int i = 0; i < POW2(PATCH_SIZE); i++)
				{
					pixF[i] = 0.0f;
				}


				// for each weight in the map
				for (int y_g = 0; y_g < wMapWidth; y_g++)
				{
					for (int x_g = 0; x_g < wMapWidth; x_g++)
					{

						// Normalize Weight
						w[y_g * wMapWidth + x_g] /= wSum;

						// For the patch regarding this weight
						// Create Pixel Array of new patch
						float* pixG = (float*)malloc(patchSizeFloat);
						for (int i = 0; i < PATCH_SIZE; i++)
							memcpy(
								&pixG[i * PATCH_SIZE],
								&img.pixelArr[(i + y_g) * img.width + x_g],
								PATCH_SIZE * sizeof(float));


						// for Each pixel in the patch
						for (int i = 0; i < POW2(PATCH_SIZE); i++)
						{
							pixF[i] += w[y_g * wMapWidth + x_g] * pixG[i];
						}

					}
				}


				// Copy Corrected Pixels into result array
				for (int i = 0; i < PATCH_SIZE; i++)
				{
					for (int j = 0; j < PATCH_SIZE; j++)
					{
						pix[(y_f + i) * img.width + x_f + j] = pixF[i * PATCH_SIZE + j];
					}
				}


				//if (pixF != NULL)
				//{
				free(pixF);
				pixF = NULL;
				//}

				//if (w != NULL)
				//{
				free(w);
				w = NULL;
				//}

			}

		}

		cout << "CPU Took" << clock.stopClock() << "\n";


#ifdef USE_LOG_FILE

		/*log << "\n\n";

		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				log << pix[i * 16 + j] << " ";
			}
			log << "\n";
		}*/

		log.close();
#endif

		memcpy(&img.pixelArr[0], &pix[0], POW2(img.width) * sizeof(float));

		img.Write(params.input.outputDir + "/CPU_sigma" + to_string(params.algorithm.sigma) + "_" + utils::ImageFile::GetFileName(params.input.imgPath));

		return 0;
	}

}

