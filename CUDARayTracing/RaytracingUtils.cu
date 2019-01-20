#include "RaytracingUtils.h"

#include "Camera.h"
#include "Ray.h"
#include "Vector3.h"

namespace
{
	__device__ Vector3 backgroundColor(const Ray& ray)
	{
		Vector3 unit_direction = normalize(ray.direction());
		float t = 0.5f*(unit_direction.y() + 1.0f);
		return (1.0f - t)*Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
	}

	__global__ void renderColors(Vector3* colors, Camera* camera, int pixelsWidth, int pixelsHeight)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) return;
		int pixel_index = j * pixelsWidth + i;
		float u = float(i) / float(pixelsWidth);
		float v = float(j) / float(pixelsHeight);
		colors[pixel_index] = backgroundColor(camera->GenerateRay(u, v));
	}
}

namespace RaytracingUtils
{
	__host__ void getColors(Vector3* colors, Camera* camera, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
	{
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		renderColors<<<blocks, threads>>>(colors, camera, pixelsWidth, pixelsHeight);
	}
}