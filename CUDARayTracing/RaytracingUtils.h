#ifndef RAYTRACINGUTILS_H
#define RAYTRACINGUTILS_H

#include "Cuda.h"

class Entity;
class Camera;
class Vector3;

namespace RaytracingUtils
{
	__host__ void getColors(Vector3* colors, Entity** entities, int numberOfEntities, Camera* camera, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY);

	__host__ void initWorld(Entity** list, Entity** d_entities);
	__host__ void cleanUpWorld(Entity** list, Entity** d_entities);
}

#endif // !RAYTRACINGUTILS_H
