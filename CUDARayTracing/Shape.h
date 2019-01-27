#ifndef SHAPE_H
#define SHAPE_H

#include "HitInfo.h"
#include "Ray.h"

class Shape
{
public:
	__device__ virtual bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const = 0;
};

class Sphere : public Shape
{
public:
	__device__ Sphere() = default;
	__device__ Sphere(const Vector3& center, float radius) : _center(center), _radius(radius) {}
	__device__ bool Hit(const Ray& ray, float minDist, float maxDist, HitInfo& hitInfo) const
	{
		Vector3 oc = ray.pos - _center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(oc, ray.dir);
		float c = dot(oc, oc) - _radius * _radius;
		float discriminant = b * b - a * c;
		if (discriminant <= 0.0f)
		{
			//No real solution -> no hit
			return false;
		}

		float squaredDiscriminant = sqrt(discriminant);
		float negativeRoot = (-b - squaredDiscriminant) / a;
		if (negativeRoot < maxDist && negativeRoot > minDist)
		{
			hitInfo.isHit = true;
			hitInfo.distance = negativeRoot;
			hitInfo.point = ray.getPoint(negativeRoot);
			hitInfo.normal = (hitInfo.point - _center) / _radius;
			return true;
		}

		float positiveRoot = (-b + squaredDiscriminant) / a;
		if (positiveRoot < maxDist && positiveRoot > minDist)
		{
			hitInfo.isHit = true;
			hitInfo.distance = positiveRoot;
			hitInfo.point = ray.getPoint(positiveRoot);
			hitInfo.normal = (hitInfo.point - _center) / _radius;
			return true;
		}

		return false;
	}

private:
	Vector3 _center;
	float _radius;
};

#endif // !SHAPE_H
