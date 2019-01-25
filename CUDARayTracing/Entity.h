#ifndef ENTITY_H
#define ENTITY_H

#include "HitInfo.h"
#include "Managed.h"
#include "Ray.h"
#include "Shape.h"

class Entity
{
public:
	__device__ Entity(Shape* shape) : _shape(shape) {}

	__device__ ~Entity()
	{
		delete _shape;
	}

	__device__ bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const
	{
		return _shape->Hit(ray, minDist, maxDist, hitInfo);
	}

private:
	Shape* _shape = nullptr;
};

#endif // !ENTITY_H
