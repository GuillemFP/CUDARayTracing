#ifndef ENTITY_H
#define ENTITY_H

#include "HitInfo.h"
#include "Managed.h"
#include "Ray.h"

class Entity
{
public:
	__device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const = 0;
};

#endif // !ENTITY_H
