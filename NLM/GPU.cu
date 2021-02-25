/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

#include "GPU.cuh"

using namespace std;

using std::cout;
using std::generate;
using std::vector;

// Threads per CTA dimension
constexpr int _THREADS = 1 << 5;

// Size of shared memory per Thread Block
constexpr int _SHMEM_SIZE = _THREADS * _THREADS;




int main2(int argc, char** argv)
{

	int _n = 0;
	int _threads = _THREADS;
	int _blocks = (_n + _threads - 1) / _threads;

	dim3 _dim_blocks(_blocks, _blocks);
	dim3 _dim_threads(_threads, _threads);

	krnl <<<_dim_blocks, _dim_threads >>> ();
	cudaDeviceSynchronize();



	return EXIT_SUCCESS;

}



__global__ void krnl()
{
	int row = threadIdx.x * blockDim.x + threadIdx.x;
	int col = threadIdx.y * blockDim.y + threadIdx.y;


	return;
}
