#ifndef RAYTRACINGUTILS_H
#define RAYTRACINGUTILS_H

#include "Cuda.h"
#include <curand_kernel.h>

class Entity;
class EntityList;
class Camera;
class Vector3;
class Screen;

namespace RaytracingUtils
{
	__host__ void getColors(Screen* screen, EntityList** entities, Camera* camera, curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY);

    __host__ void initRender(curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY);
	__host__ void initEntities(EntityList** entities);
	__host__ void cleanUpEntities(EntityList** entities);
}

namespace MathUtils
{
	__device__ Vector3 RandomPointInSphere(curandState* rand);
}

#endif // !RAYTRACINGUTILS_H
