#ifndef SHAPE_H
#define SHAPE_H

#include "HitInfo.h"
#include "Managed.h"
#include "Ray.h"

class Shape
{
public:
	__device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const = 0;
};

#endif // !SHAPE_H
