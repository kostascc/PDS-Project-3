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

		// Start Clock
		utils::Clock clock = utils::Clock();
		clock.startClock();

		utils::ImageFile img = utils::ImageFile();
		img.Read(params.input.imgPath);


#ifdef USE_LOG_FILE
		ofstream log;
		log.open("./livelog_CPU.txt");
#endif


		int patchSizeFloat = params.algorithm.patchSize * params.algorithm.patchSize * sizeof(float);

		float* pix = (float*)calloc(pow(img.width, 2), sizeof(float));

		float sigmaSquared = pow(params.algorithm.sigma, 2);

		// Width of weight matrix
		int wMapWidth = img.width - params.algorithm.patchSize;

		// For Each patch of size 'patchSize'


#ifdef USE_OPENMP
#pragma omp parallel for
#endif
		for (int y_f = 0; y_f < img.width - params.algorithm.patchSize; y_f++)
		{
			log << "\n";
			cout << "y " << y_f << "\n";
			for (int x_f = 0; x_f < img.width - params.algorithm.patchSize; x_f++)
			{
				// x_f, y_f: Coordinates of patch starting pixel
				// x: Width
				// y: height



				// Create Pixel Matrix of Patch
				float* pixF = (float*)malloc(patchSizeFloat);
				for (int i = 0; i < params.algorithm.patchSize; i++)
					memcpy(
						&pixF[i * params.algorithm.patchSize],
						&img.pixelArr[(i + y_f) * img.width + x_f],
						params.algorithm.patchSize * sizeof(float));


				// Sum of weights
				double wSum = 0.0f;


				// Create Weight map
				double* w = (double*)calloc(pow(wMapWidth, 2), sizeof(double));


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
						for (int i = 0; i < params.algorithm.patchSize; i++)
							memcpy(
								&pixG[i * params.algorithm.patchSize],
								&img.pixelArr[(i + y_g) * img.width + x_g],
								params.algorithm.patchSize * sizeof(float));


						// Find Euclidean Distance squared
						double d = 0.0f;
						for (int i = 0; i < pow(params.algorithm.patchSize, 2); i++)
						{
							d += pow((pixF[i] - pixG[i]), 2);
						}

						d = exp(-d / (sigmaSquared));

						// Calculate non-normalized weight
						w[y_g * wMapWidth + x_g] = d;
						wSum += d;


						free(pixG);
						pixG = NULL;

					}
				}



				// Clean pixels in batch
				for (int i = 0; i < pow(params.algorithm.patchSize, 2); i++)
				{
					pixF[i] = 0.0f;
				}


				log << wSum << " ";


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
						for (int i = 0; i < params.algorithm.patchSize; i++)
							memcpy(
								&pixG[i * params.algorithm.patchSize],
								&img.pixelArr[(i + y_g) * img.width + x_g],
								params.algorithm.patchSize * sizeof(float));


						// for Each pixel in the patch
						for (int i = 0; i < pow(params.algorithm.patchSize, 2); i++)
						{
							pixF[i] += w[y_g * wMapWidth + x_g] * pixG[i];
						}

					}
				}


				// Copy Corrected Pixels into result array
				for (int i = 0; i < params.algorithm.patchSize; i++)
				{
					for (int j = 0; j < params.algorithm.patchSize; j++)
					{
						pix[(y_f + i) * img.width + x_f + j] = pixF[i * params.algorithm.patchSize + j];
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


#ifdef USE_LOG_FILE

		log << "\n\n";

		/*for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				log << pix[i * 16 + j] << " ";
			}
			log << "\n";
		}*/

		log.close();
#endif

		//memcpy(&img.pixelArr[0], &pix[0], pow(img.width, 2) * sizeof(float));


		img.Write(params.input.outputDir + "/CPU_sigma" + to_string(params.algorithm.sigma) + "_" + utils::ImageFile::GetFileName(params.input.imgPath));


		cout << "CPU Took" << clock.stopClock() << "\n";

		return 0;
	}

}

