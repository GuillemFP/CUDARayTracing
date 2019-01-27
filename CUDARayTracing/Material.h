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

class Metal : public Material
{
public:
	__device__ Metal(const Vector3& albedo, float fuzziness) : _albedo(albedo), _fuzziness(fuzziness) {}
	__device__ bool Scatter(const Ray& ray, const HitInfo& hitInfo, ScatterInfo& scatterInfo, curandState* rand) const
	{
		scatterInfo.scatteredRay.pos = hitInfo.point;
		scatterInfo.scatteredRay.time = ray.time;

		Vector3 scatteredRay = MathUtils::ReflectedVector(ray.dir, hitInfo.normal) + _fuzziness * MathUtils::RandomPointInSphere(rand);
		scatterInfo.scatteredRay.dir = normalize(scatteredRay);
		scatterInfo.attenuation = _albedo;

		scatterInfo.scatters = dot(scatterInfo.scatteredRay.dir, hitInfo.normal) > 0;
		return scatterInfo.scatters;
	}

private:
	Vector3 _albedo;
	float _fuzziness = 0.0f;
};

class Dielectric : public Material
{
public:
	__device__ Dielectric(float refractiveIndex) : _refractiveIndex(refractiveIndex) {}
	__device__ bool Scatter(const Ray& ray, const HitInfo& hitInfo, ScatterInfo& scatterInfo, curandState* rand) const
	{
		Vector3 normal;
		float refractionFactorRatio;

		// Positive value means ray crossing dielectric from inside to outside
		if (dot(ray.dir, hitInfo.normal) > 0)
		{
			normal = -hitInfo.normal;
			refractionFactorRatio = _refractiveIndex;
		}
		else
		{
			normal = hitInfo.normal;
			refractionFactorRatio = 1.0f / _refractiveIndex;
		}
		float cosine = MathUtils::CosineIncidentAngle(normal, ray.dir);

		Vector3 refracted;
		scatterInfo.attenuation = Vector3(1.0f, 1.0f, 1.0f);
		if (MathUtils::Refracts(ray.dir, normal, refractionFactorRatio, refracted))
		{
			float reflectionCoefficient = MathUtils::SchlickApproximation(refractionFactorRatio, cosine);
			if (curand_uniform(rand) >= reflectionCoefficient)
			{
				scatterInfo.scatters = true;
				scatterInfo.scatteredRay = Ray(hitInfo.point, refracted, ray.time);
				return true;
			}
		}

		scatterInfo.scatters = true;
		scatterInfo.scatteredRay = Ray(hitInfo.point, MathUtils::ReflectedVector(ray.dir, hitInfo.normal), ray.time);
		return true;
	}

private:
	float _refractiveIndex = 0.0f;
};

#endif // !MATERIAL_H
