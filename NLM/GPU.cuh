#ifndef GPU_CUH
#define GPU_CUH


#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include "Utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "math.h"
#include <omp.h>


#define USE_LOG_FILE


using namespace std;
using namespace parameters;

namespace GPU
{

	int run(Parameters params);

}


#endif
