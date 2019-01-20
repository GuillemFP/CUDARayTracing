#ifndef RAY_H
#define RAY_H

#include "Vector3.h"
#include "Cuda.h"

class Ray
{
public:
	__host__ __device__ Ray() {}
	__host__ __device__ Ray(const Vector3& position, const Vector3& direction, const float time = 0.0f) : pos(position), dir(direction), time(time) {}
	__host__ __device__ Vector3 origin() const { return pos; }
	__host__ __device__ Vector3 direction() const { return dir; }
	__host__ __device__ Vector3 getPoint(float t) const { return pos + t*dir; }

	Vector3 pos;
	Vector3 dir;
	float time = 0.0f;
};

#endif // !RAY_H