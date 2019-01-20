#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include "Cuda.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) CudaUtils::checkErrors( (val), #val, __FILE__, __LINE__ )

namespace CudaUtils
{
	__host__ void checkErrors(cudaError_t result, char const *const func, const char *const file, int const line);
}

#endif // !CUDAUTILS_H
