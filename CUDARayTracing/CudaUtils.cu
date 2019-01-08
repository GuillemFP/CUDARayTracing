#include "CudaUtils.h"

#include <iostream>

namespace
{
	__global__ void renderColors(Vector3* colors, int pixelsWidth, int pixelsHeight)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) return;
		int pixel_index = j * pixelsWidth + i;
		colors[pixel_index] = Vector3(float(i) / pixelsWidth, float(j) / pixelsHeight, 0.2f);
	}
}

namespace CudaUtils
{
	__host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
	{
		if (result) {
			std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
				file << ":" << line << " '" << func << "' \n";
			// Make sure we call CUDA Device Reset before exiting
			cudaDeviceReset();
			exit(99);
		}
	}

	__host__ void getColors(Vector3* colors, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
	{
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		renderColors<<<blocks, threads>>>(colors, pixelsWidth, pixelsHeight);
	}
}
