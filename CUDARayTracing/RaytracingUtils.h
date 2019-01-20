#ifndef RAYTRACINGUTILS_H
#define RAYTRACINGUTILS_H

#include "Cuda.h"

class Camera;
class Vector3;

namespace RaytracingUtils
{
	__host__ void getColors(Vector3* colors, Camera* camera, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY);
}

#endif // !RAYTRACINGUTILS_H
