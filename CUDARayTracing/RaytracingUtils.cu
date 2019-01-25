#include "RaytracingUtils.h"

#include "Entity.h"
#include "EntityList.h"
#include "Sphere.h"
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

	__device__ Vector3 color(const Ray& ray, const EntityList** entities)
	{
		HitInfo hitInfo;
		if((*entities)->Hit(ray, 0.0f, FLT_MAX, hitInfo))
		{
			return 0.5f * Vector3(hitInfo.normal.x() + 1.0f, hitInfo.normal.y() + 1.0f, hitInfo.normal.z() + 1.0f);
		}
		else
		{
			return backgroundColor(ray);
		}
	}

	__global__ void renderColors(Vector3* colors, EntityList** entities, Camera* camera, int pixelsWidth, int pixelsHeight)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) return;
		int pixel_index = j * pixelsWidth + i;
		float u = float(i) / float(pixelsWidth);
		float v = float(j) / float(pixelsHeight);
		colors[pixel_index] = color(camera->GenerateRay(u, v), entities);
	}

	__global__ void createEntities(EntityList** entities) 
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) 
		{
			*(entities) = new EntityList(2);
			(*entities)->push_back(new Entity(new Sphere(Vector3(0, 0, -1), 0.5)));
			(*entities)->push_back(new Entity(new Sphere(Vector3(0, -100.5, -1), 100)));
		}
	}

	__global__ void freeEntities(EntityList** entities)
	{
		delete *entities;
	}
}

namespace RaytracingUtils
{
	__host__ void getColors(Vector3* colors, EntityList** entities, Camera* camera, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
	{
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		renderColors<<<blocks, threads>>>(colors, entities, camera, pixelsWidth, pixelsHeight);
	}

	__host__ void initEntities(EntityList** entities)
	{
		createEntities<<<1, 1>>>(entities);
	}

	__host__ void cleanUpEntities(EntityList** entities)
	{
		freeEntities<<<1, 1 >>>(entities);
	}
}