#ifndef RAYTRACINGUTILS_H
#define RAYTRACINGUTILS_H

#include "Cuda.h"

class Entity;
class EntityList;
class Camera;
class Vector3;

namespace RaytracingUtils
{
	__host__ void getColors(Vector3* colors, EntityList** entities, Camera* camera, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY);

	__host__ void initEntities(EntityList** entities);
	__host__ void cleanUpEntities(EntityList** entities);
}

#endif // !RAYTRACINGUTILS_H
