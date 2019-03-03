#include "RaytracingUtils.h"

#include "EntityList.h"
#include "EntityData.h"
#include "Shape.h"
#include "Camera.h"
#include "Screen.h"

#define SEED 1984
#define RANDBLOCKSIZE 10000

namespace
{
	__device__ Vector3 background_color(const Ray& ray)
	{
		Vector3 unit_direction = normalize(ray.direction());
		float t = 0.5f*(unit_direction.y() + 1.0f);
		return (1.0f - t)*Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0);
	}

	__device__ Vector3 color(const Ray& ray, const EntityList** entities, curandState* rand, int scatters)
	{
		Ray propagatedRay = ray;
		Vector3 totalAttenuation = Vector3(1.0f, 1.0f, 1.0f);
		float cudaAtt = 1.0f;

		for (int i = 0; i < scatters; ++i)
		{
			HitInfo hitInfo;
			if ((*entities)->Hit(propagatedRay, 0.001f, FLT_MAX, hitInfo))
			{
				ScatterInfo scatterInfo;
				if (hitInfo.entity->Scatter(propagatedRay, hitInfo, scatterInfo, rand))
				{
					totalAttenuation *= scatterInfo.attenuation;
					propagatedRay = scatterInfo.scatteredRay;
				}
				else
				{
					return Vector3(0.0f, 0.0f, 0.0f);
				}
			}
			else
			{
				return totalAttenuation * background_color(propagatedRay);
			}
		}

		return Vector3(0.0f, 0.0f, 0.0f);
	}

	__device__ Shape* createShape(const ShapeData& data)
	{
		switch (data.type)
		{
		case ShapeType::Sphere:
		{
			return new Sphere(*data.position, data.radius);
		}
		default:
			break;
		}

		return nullptr;
	}

	__device__ Material* createMaterial(const MaterialData& data)
	{
		switch (data.type)
		{
		case MaterialType::Diffuse:
			return new Lambertian(*data.color);
		case MaterialType::Metal:
			return new Metal(*data.color, data.fuzziness);
		case MaterialType::Dielectric:
			return new Dielectric(data.refractiveIndex);
		}

		return nullptr;
	}

	__device__ Entity* createEntity(const EntityData& data)
	{
		Shape* shape = createShape(*data.shapeData);
		Material* material = createMaterial(*data.materialData);
		if (shape && material)
		{
			return new Entity(shape, material);
		}

		return nullptr;
	}

	__global__ void render_colors(Screen* screen, EntityList** entities, Camera* camera, curandState* randStates, int pixelsWidth, int pixelsHeight, int scatters)
	{
		const int i = threadIdx.x + blockIdx.x * blockDim.x;
		const int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) 
            return;

		const int pixelIndex = j * pixelsWidth + i;
        curandState& randState = randStates[pixelIndex];

		const float u = (float(i + curand_uniform(&randState))) / float(pixelsWidth);
		const float v = (float(j + curand_uniform(&randState))) / float(pixelsHeight);

		screen->AddColor(color(camera->GenerateRay(u, v, &randState), entities, &randState, scatters), i, j);
	}

	__global__ void init_render(curandState* randStates, int pixelsWidth, int pixelsHeight)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if ((i >= pixelsWidth) || (j >= pixelsHeight)) 
			return;
		int index = j * pixelsWidth + i;

		curand_init((SEED << 20) + index, 0, 0, &randStates[index]);
	}

	__global__ void create_entities(EntityList** entities, const EntityData* data, int numEntities)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) 
		{
			*(entities) = new EntityList(numEntities);
			for (size_t i = 0; i < numEntities; i++)
			{
				Entity* entity = createEntity(data[i]);
				if (entity)
				{
					(*entities)->push_back(entity);
				}
			}
		}
	}

	__global__ void free_entities(EntityList** entities)
	{
		delete *entities;
	}
}

namespace RaytracingUtils
{
	__host__ void getColors(Screen* screen, EntityList** entities, Camera* camera, curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY, int scatters)
	{
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		render_colors<<<blocks, threads>>>(screen, entities, camera, randStates, pixelsWidth, pixelsHeight, scatters);

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		screen->AddSample();
	}

    __host__ void initRender(curandState* randStates, int pixelsWidth, int pixelsHeight, int threadsX, int threadsY)
    {
		dim3 blocks(pixelsWidth / threadsX + 1, pixelsHeight / threadsY + 1);
		dim3 threads(threadsX, threadsY);
		init_render<<<blocks, threads>>>(randStates, pixelsWidth, pixelsHeight);
    }

	__host__ void initEntities(EntityList** entities, const EntityData* data, int numEntities)
	{
		create_entities<<<1, 1>>>(entities, data, numEntities);
	}

	__host__ void cleanUpEntities(EntityList** entities)
	{
		free_entities<<<1, 1 >>>(entities);
	}
}

namespace MathUtils
{
	__device__ Vector3 RandomPointInSphere(curandState* rand)
	{
		Vector3 ret;
		do
		{
			ret = 2.0f * Vector3(curand_uniform(rand), curand_uniform(rand), curand_uniform(rand)) - Vector3(1.0f, 1.0f, 1.0f);
		} while (ret.lengthSq() >= 1.0f);
		return ret;
	}

	__device__ Vector3 RandomPointInDisk(curandState * rand)
	{
		Vector3 ret;
		do
		{
			ret = 2.0f * Vector3(curand_uniform(rand), curand_uniform(rand), 0.0f) - Vector3(1.0f, 1.0f, 0.0f);
		} while (dot(ret, ret) >= 1.0f);
		return ret;
	}

	__device__ Vector3 ReflectedVector(const Vector3 & inVector, const Vector3 & normal)
	{
		return inVector - 2.0f * dot(inVector, normal) * normal;
	}

	__device__ float CosineIncidentAngle(const Vector3& normal, const Vector3& inVector)
	{
		return -dot(normal, inVector);
	}

	//Snell's law vectorial form
	// v_refract = r v + (r c - sqrt(1 - r^2 (1 - c^2))) n
	// r = n1/n2, c = - n * v
	__device__ bool Refracts(const Vector3& inVector, const Vector3& normal, float refractionFactorRatio, Vector3& refracted)
	{
		float c = CosineIncidentAngle(normal, inVector);
		float discriminant = 1 - refractionFactorRatio * refractionFactorRatio * (1 - c * c);
		if (discriminant < 0)
		{
			//Total internal reflection
			return false;
		}

		refracted = refractionFactorRatio * inVector + (refractionFactorRatio * c - sqrt(discriminant)) * normal;
		return true;
	}

	//Approximates reflection coefficient as function of incident angle
	__device__ float SchlickApproximation(float refractionFactorRatio, float cosine)
	{
		float r0 = (refractionFactorRatio - 1.0f) / (refractionFactorRatio + 1.0f);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * powf(1.0f - cosine, 5);
	}
}