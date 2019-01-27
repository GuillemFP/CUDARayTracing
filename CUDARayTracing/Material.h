#ifndef MATERIAL_H
#define MATERIAL_H

#include "HitInfo.h"
#include "ScatterInfo.h"
#include "RaytracingUtils.h"

class Material
{
public:
	__device__ virtual bool Scatter(const Ray& ray, const HitInfo& hitInfo, ScatterInfo& scatterInfo, curandState* rand) const = 0;
};

class Lambertian : public Material
{
public:
	__device__ Lambertian(const Vector3& albedo) : _albedo(albedo) {}
	__device__ bool Scatter(const Ray& ray, const HitInfo& hitInfo, ScatterInfo& scatterInfo, curandState* rand) const
	{
		scatterInfo.scatters = true;

		Vector3 sphereTarget = hitInfo.point + hitInfo.normal + MathUtils::RandomPointInSphere(rand);
		scatterInfo.scatteredRay.pos = hitInfo.point;
		scatterInfo.scatteredRay.dir = normalize(sphereTarget - hitInfo.point);
		scatterInfo.scatteredRay.time = ray.time;
		scatterInfo.attenuation = _albedo;

		return true;
	}

private:
	Vector3 _albedo;
};

#endif // !MATERIAL_H
