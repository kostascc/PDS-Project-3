#include <iostream>
#include "parameters.h"
#include <string.h>
#include <string>
#include "CPU.h"

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
			params.input.imgPath = (i<argc)? string(argv[i]) : "";
		}

		else if ((string(argv[i]) == "-e")) {
			i++;
			params.input.exportPath = (i<argc)? string(argv[i]) : "";
		}

	}


	(params.CPU.run) ? CPU::run(params) : true;
	(params.GPU.run) ? printf("GPU not ready yet!\n") : true;
	

	return 0;

}