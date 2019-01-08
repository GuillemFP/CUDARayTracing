#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include "Cuda.h"
#include "Vector3.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) CudaUtils::check_cuda( (val), #val, __FILE__, __LINE__ )

namespace CudaUtils
{
	__host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

	__host__ void getColors(Vector3* colors, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY);
}

#endif // !CUDAUTILS_H
