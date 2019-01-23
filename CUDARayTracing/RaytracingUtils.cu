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

	__device__ bool getHit(const Ray& ray, float minDist, float maxDist, const Entity** entities, int numberOfEntities, HitInfo& hitInfo)
	{
		HitInfo currentHitInfo;
		float currentMaxDistance = maxDist;
		for (int i = 0; i < numberOfEntities; i++)
		{
			if (entities[i]->Hit(ray, minDist, currentMaxDistance, currentHitInfo))
			{
				currentMaxDistance = currentHitInfo.distance;
				hitInfo = currentHitInfo;
			}
		}

		return hitInfo.isHit;
	}

	__device__ Vector3 color(const Ray& ray, const Entity** entities, int numberOfEntities)
	{
		HitInfo hitInfo;
		//if (getHit(ray, 0.0f, FLT_MAX, entities, numberOfEntities, hitInfo))
		if((*entities)->Hit(ray, 0.0f, FLT_MAX, hitInfo))
		{
			return 0.5f * Vector3(hitInfo.normal.x() + 1.0f, hitInfo.normal.y() + 1.0f, hitInfo.normal.z() + 1.0f);
		}
		else
		{
			return backgroundColor(ray);
		}
	}

	__global__ void renderColors(Vector3* colors, Entity** entities, int numberOfEntities, Camera* camera, int pixelsWidth, int pixelsHeight)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) return;
		int pixel_index = j * pixelsWidth + i;
		float u = float(i) / float(pixelsWidth);
		float v = float(j) / float(pixelsHeight);
		colors[pixel_index] = color(camera->GenerateRay(u, v), entities, numberOfEntities);
	}

	__global__ void createWorld(Entity** list, Entity** d_entities) 
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) 
		{
			*(list) = new Sphere(Vector3(0, 0, -1), 0.5);
			*(list + 1) = new Sphere(Vector3(0, -100.5, -1), 100);
			*d_entities = new EntityList(list, 2);
		}
	}

	__global__ void freeWorld(Entity **list, Entity **d_entities)
	{
		delete *(list);
		delete *(list + 1);
		delete *d_entities;
	}
}

namespace RaytracingUtils
{
	__host__ void getColors(Vector3* colors, Entity** entities, int numberOfEntities, Camera* camera, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
	{
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		renderColors<<<blocks, threads>>>(colors, entities, numberOfEntities, camera, pixelsWidth, pixelsHeight);
	}

	__host__ void initWorld(Entity** list, Entity** d_entities)
	{
		createWorld<<<1, 1>>>(list, d_entities);
	}

	__host__ void cleanUpWorld(Entity** list, Entity** d_entities)
	{
		freeWorld <<<1, 1 >>>(list, d_entities);
	}
}