#ifndef PARSEUTILS_H
#define PARSEUTILS_H

#include "Vector3.h"
#include "EntityData.h"

class Config;
class ConfigArray;

namespace ParseUtils
{
	Vector3 ParseVector(const ConfigArray& config, const Vector3& defaultValue = Vector3(1.0f, 1.0f, 1.0f));

	ShapeType ParseShapeTypeFromString(const std::string& str);
	MaterialType ParseMaterialTypeFromString(const std::string& str);

	void ParseEntity(EntityData& entityData, const Config& entity);
	void ParseShape(ShapeData& shapeData, const Config& shape);
	void ParseMaterial(MaterialData& materialData, const Config& material);
}

#endif // !PARSEUTILS_H