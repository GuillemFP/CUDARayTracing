#ifndef ENTITYDATA_H
#define ENTITYDATA_H

#include "Managed.h"
#include "Vector3.h"

enum class ShapeType
{
	Sphere = 0,
	Unknown
};

enum class MaterialType
{
	Diffuse = 0,
	Metal,
	Dielectric,
	Unknown
};

class ShapeData : public Managed
{
public:
	ShapeData()
	{
		checkCudaErrors(cudaMallocManaged(&position, sizeof(Vector3)));
	}

	~ShapeData()
	{
		checkCudaErrors(cudaFree(position));
	}

	ShapeType type;
	Vector3* position = nullptr;
	float radius = 1.0f;
};

class MaterialData : public Managed
{
public:
	MaterialData()
	{
		checkCudaErrors(cudaMallocManaged(&color, sizeof(Vector3)));
	}

	~MaterialData()
	{
		checkCudaErrors(cudaFree(color));
	}

	MaterialType type;
	Vector3* color = nullptr;
	float fuzziness = 0.0f;
	float refractiveIndex = 1.0f;
};

class EntityData : public Managed
{
public:
	EntityData()
	{
		shapeData = new ShapeData();
		materialData = new MaterialData();
	}

	~EntityData()
	{
		checkCudaErrors(cudaFree(shapeData));
		checkCudaErrors(cudaFree(materialData));
	}

	ShapeData* shapeData = nullptr;
	MaterialData* materialData = nullptr;
};

#endif // !ENTITYDATA_H
