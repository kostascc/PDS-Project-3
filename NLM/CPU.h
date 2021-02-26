/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/


#ifndef CPU_H
#define CPU_H

#include "cuda_runtime.h"
#include "Parameters.h"
#include "Utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "math.h"
#include <omp.h>


#define USE_LOG_FILE
#define LOG_W_SUMS

//#define USE_OPENMP


using namespace std;
using namespace parameters;

namespace CPU
{

	int run(Parameters params);

}

#endif
