#include "RaytracingUtils.h"

#include "EntityList.h"
#include "Shape.h"
#include "Camera.h"
#include "Screen.h"

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

	__device__ Vector3 color(const Ray& ray, const EntityList** entities, curandState* rand)
	{
		Ray propagatedRay = ray;
		Vector3 totalAttenuation = Vector3(1.0f, 1.0f, 1.0f);
		float cudaAtt = 1.0f;

		for (int i = 0; i < 50; ++i)
		{
			HitInfo hitInfo;
			if ((*entities)->Hit(propagatedRay, 0.001f, FLT_MAX, hitInfo))
			{
				ScatterInfo scatterInfo;
				if (hitInfo.entity->Scatter(propagatedRay, hitInfo, scatterInfo, rand))
				{
					totalAttenuation *= scatterInfo.attenuation;
					propagatedRay = scatterInfo.scatteredRay;
				}
				else
				{
					return Vector3(0.0f, 0.0f, 0.0f);
				}
			}
			else
			{
				return totalAttenuation * background_color(propagatedRay);
			}
		}

		return Vector3(0.0f, 0.0f, 0.0f);
	}

	__global__ void render_colors(Screen* screen, EntityList** entities, Camera* camera, curandState* randStates, int pixelsWidth, int pixelsHeight)
	{
		const int i = threadIdx.x + blockIdx.x * blockDim.x;
		const int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) 
            return;

		const int pixelIndex = j * pixelsWidth + i;
        curandState& randState = randStates[pixelIndex];

		const float u = (float(i + curand_uniform(&randState))) / float(pixelsWidth);
		const float v = (float(j + curand_uniform(&randState))) / float(pixelsHeight);

		screen->AddColor(color(camera->GenerateRay(u, v), entities, &randState), i, j);
	}

	__global__ void init_render(curandState* randStates, int pixelsWidth, int pixelsHeight)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) 
			return;
		int index = j * pixelsWidth + i;

		curand_init((SEED << 20) + index, 0, 0, &randStates[index]);
	}

	__global__ void create_entities(EntityList** entities)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) 
		{
			*(entities) = new EntityList(2);
			(*entities)->push_back(new Entity(new Sphere(Vector3(0.0f, 0.0f, -1.0f), 0.5f), new Lambertian(Vector3(0.5f, 0.5f, 0.5f))));
			(*entities)->push_back(new Entity(new Sphere(Vector3(0.0f, -100.5f, -1.0f), 100.0f), new Lambertian(Vector3(0.5f, 0.5f, 0.5f))));
		}
	}

	__global__ void free_entities(EntityList** entities)
	{
		delete *entities;
	}
}

namespace RaytracingUtils
{
	__host__ void getColors(Screen* screen, EntityList** entities, Camera* camera, curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
	{
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		render_colors<<<blocks, threads>>>(screen, entities, camera, randStates, pixelsWidth, pixelsHeight);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		screen->AddSample();
	}

    __host__ void initRender(curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
    {
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		init_render<<<blocks, threads>>>(randStates, pixelsWidth, pixelsHeight);
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

namespace MathUtils
{
	__device__ Vector3 RandomPointInSphere(curandState* rand)
	{
		Vector3 ret;
		do
		{
			ret = 2.0f * Vector3(curand_uniform(rand), curand_uniform(rand), curand_uniform(rand)) - Vector3(1.0f, 1.0f, 1.0f);
		} while (ret.lengthSq() >= 1.0f);
		return ret;
	}
}