#include "CudaUtils.h"

#include <iostream>

namespace CudaUtils
{
	__host__ void checkErrors(cudaError_t result, char const *const func, const char *const file, int const line)
	{
		if (result) {
			std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
				file << ":" << line << " '" << func << "' \n";
			// Make sure we call CUDA Device Reset before exiting
			cudaDeviceReset();
			exit(99);
		}
	}
}
