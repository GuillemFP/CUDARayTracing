#include "RaytracingUtils.h"

#include "Entity.h"
#include "EntityList.h"
#include "Sphere.h"
#include "Camera.h"
#include "Ray.h"
#include "Vector3.h"

#define SEED 1984
#define RANDBLOCKSIZE 10000

namespace
{
	__device__ Vector3 background_color(const Ray& ray)
	{
		Vector3 unit_direction = normalize(ray.direction());
		float t = 0.5f*(unit_direction.y() + 1.0f);
		return (1.0f - t)*Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
	}

	__device__ Vector3 color(const Ray& ray, const EntityList** entities)
	{
		HitInfo hitInfo;
		if((*entities)->Hit(ray, 0.0f, FLT_MAX, hitInfo))
		{
			return 0.5f * Vector3(hitInfo.normal.x() + 1.0f, hitInfo.normal.y() + 1.0f, hitInfo.normal.z() + 1.0f);
		}
		else
		{
			return background_color(ray);
		}
	}

	__global__ void render_colors(Vector3* colors, EntityList** entities, Camera* camera, curandState* randStates, int pixelsWidth, int pixelsHeight)
	{
		const int i = threadIdx.x + blockIdx.x * blockDim.x;
		const int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) 
            return;

		const int pixelIndex = j * pixelsWidth + i;
        curandState randState = randStates[pixelIndex];

		const float rand1 = curand_uniform(&randState);
		const float rand2 = curand_uniform(&randState);

		const float u = (float(i + rand1)) / float(pixelsWidth);
		const float v = (float(j + rand2)) / float(pixelsHeight);
		colors[pixelIndex] += color(camera->GenerateRay(u, v), entities);
	}

    __global__ void init_render(curandState* randStates, int totalNumber, int i)
    {
		const int index = i * RANDBLOCKSIZE + threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= totalNumber)
			return;

		curand_init((SEED << 20) + index, 0, 0, &randStates[index]);
    }

	__global__ void create_entities(EntityList** entities)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) 
		{
			*(entities) = new EntityList(2);
			(*entities)->push_back(new Entity(new Sphere(Vector3(0.0f, 0.0f, -1.0f), 0.5f)));
			(*entities)->push_back(new Entity(new Sphere(Vector3(0.0f, -100.5f, -1.0f), 100.0f)));
		}
	}

	__global__ void free_entities(EntityList** entities)
	{
		delete *entities;
	}
}

namespace RaytracingUtils
{
	__host__ void getColors(Vector3* colors, EntityList** entities, Camera* camera, curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
	{
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		render_colors<<<blocks, threads>>>(colors, entities, camera, randStates, pixelsWidth, pixelsHeight);
	}

    __host__ void initRender(curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
    {
		const int totalNumber = pixelsWidth * pixelsHeight;
		const int numIterations = totalNumber / (threadsX * RANDBLOCKSIZE) + 1;

		dim3 blocks(RANDBLOCKSIZE);
		dim3 threads(threadsX);
		for (int i = 0; i < numIterations; i++)
		{
			init_render<<<blocks, threads>>>(randStates, totalNumber, i);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
		}
    }

	__host__ void initEntities(EntityList** entities)
	{
		create_entities<<<1, 1>>>(entities);
	}

	__host__ void cleanUpEntities(EntityList** entities)
	{
		free_entities<<<1, 1 >>>(entities);
	}
}