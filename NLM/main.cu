/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

#include <iostream>
#include "parameters.h"
#include <string.h>
#include <string>
#include "CPU.h"
#include "GPU.cuh"

using namespace std;
using namespace parameters;

/**
 * @param argc Input Argument Count
 * @param argv Input Arguments
 *
 * @return 0: Success, >0: Failure
 **/
int main(int argc, char* argv[])
{

	Parameters params;

	if (argc <= 1)
	{
		cout << "Usage: \n";
		cout << " -cpu                 : Run CPU algorithm\n";
		cout << " -gpu                 : Run GPU algorithm\n";
		cout << " -t <int>             : Set CPU threads (8)\n";
		cout << " -i ./<dir>/<img>.bmp : Input Image (../in/face_128x128.bmp)\n";
		cout << " -o ./<out_dir>       : Output Directory (../out)\n";
		cout << " -s <float>           : Sigma (0.05)\n";

	}

	for (int i = 1; i < argc; i++)
	{

		if ((string(argv[i]) == "-cpu")) {
			params.CPU.run = true;
		}

		else if ((string(argv[i]) == "-gpu")) {
			params.GPU.run = true;
		}

		else if ((string(argv[i]) == "-i")) {
			i++;
			params.input.imgPath = (i < argc) ? string(argv[i]) : "";
		}

		else if ((string(argv[i]) == "-o")) {
			i++;
			params.input.outputDir = (i < argc) ? string(argv[i]) : "";
		}

		else if ((string(argv[i]) == "-t")) {
			i++;
			params.CPU.threadsMax = (i < argc) ? atoi(argv[i]) : 1;
		}

		else if ((string(argv[i]) == "-s")) {
			i++;
			params.algorithm.sigma = (i < argc) ? atof(argv[i]) : 0.05f;
		}

	}

	// Force Set OMP Threads
	omp_set_num_threads(params.CPU.threadsMax);

	// Run CPU / GPU algorithms
	(params.CPU.run) ? CPU::run(params) : true;
	(params.GPU.run) ? GPU::run(params) : true;


	return 0;

}