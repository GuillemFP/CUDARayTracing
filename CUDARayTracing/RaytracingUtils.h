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
	__host__ void getColors(Screen* screen, EntityList** entities, Camera* camera, curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY, int scatters);

    __host__ void initRender(curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY);
	__host__ void initEntities(EntityList** entities);
	__host__ void cleanUpEntities(EntityList** entities);
}

namespace MathUtils
{
	__device__ Vector3 RandomPointInSphere(curandState* rand);
	__device__ Vector3 RandomPointInDisk(curandState* rand);
	__device__ Vector3 ReflectedVector(const Vector3& inVector, const Vector3& normal);
	__device__ float CosineIncidentAngle(const Vector3& normal, const Vector3& inVector);
	__device__ bool Refracts(const Vector3& inVector, const Vector3& normal, float refractionFactorRatio, Vector3& refracted);
	__device__ float SchlickApproximation(float refractionFactorRatio, float cosine);
}

#endif // !RAYTRACINGUTILS_H
