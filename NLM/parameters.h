/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <string.h>
#include <string>
#include <time.h>

using namespace std;


#define DEBUG


namespace parameters
{

	struct Parameters
	{

		struct Algorithm
		{

			float sigma = 0.08f;

			int patchSize = 3;

		};
		Algorithm algorithm;

		struct Noise
		{
			;
		};
		Noise noise;


		struct Cpu
		{
			int threadsMax = 8;

			bool run = false;	// Run CPU algorithm
		};
		Cpu CPU;


		struct Gpu
		{
			int threadsPerBlock = 32;

			bool run = false;	// Run GPU algorithm
		};
		Gpu GPU;


		struct Input
		{
			// Image Path ./dir/img.bmp
			string imgPath = "";

			// Output Directory Path ./dir
			string outputDir = "../output";

		};
		Input input;

	};

}

#endif