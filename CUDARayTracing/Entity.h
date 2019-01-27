#ifndef ENTITY_H
#define ENTITY_H

#include "HitInfo.h"
#include "Material.h"
#include "Ray.h"
#include "Shape.h"
#include <curand_kernel.h>

class Entity
{
public:
	__device__ Entity(Shape* shape, Material* material) : _shape(shape), _material(material) {}

	__device__ ~Entity()
	{
		delete _material;
		delete _shape;
	}

	__device__ bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const
	{
		if (_shape->Hit(ray, minDist, maxDist, hitInfo))
		{
			hitInfo.entity = this;
			return true;
		}

		return false;
	}

	__device__ bool Scatter(const Ray& ray, const HitInfo& hitInfo, ScatterInfo& scatterInfo, curandState* rand) const
	{
		return _material->Scatter(ray, hitInfo, scatterInfo, rand);
	}

private:
	Shape* _shape = nullptr;
	Material* _material = nullptr;
};

#endif // !ENTITY_H
