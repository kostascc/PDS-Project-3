/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

#ifndef CPU_H
#define CPU_H

#include "Parameters.h"
#include "Utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "math.h"
#include <omp.h>


#define USE_LOG_FILE


using namespace std;
using namespace parameters;

namespace CPU
{

	int run(Parameters params);

}

#endif
