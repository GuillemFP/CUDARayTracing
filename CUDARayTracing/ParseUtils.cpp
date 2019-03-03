#include "ParseUtils.h"

#include "Config.h"
#include "Globals.h"

namespace ParseUtils
{
	Vector3 ParseVector(const ConfigArray& config, const Vector3& defaultValue)
	{
		return Vector3(config.GetFloat(0, defaultValue.x()), config.GetFloat(1, defaultValue.y()), config.GetFloat(2, defaultValue.z()));
	}

	ShapeType ParseShapeTypeFromString(const std::string & str)
	{
		if (str == "Sphere")
			return ShapeType::Sphere;

		APPLOG("Invalid entity type");

		return ShapeType::Unknown;
	}

	MaterialType ParseMaterialTypeFromString(const std::string & str)
	{
		if (str == "Diffuse")
			return MaterialType::Diffuse;
		if (str == "Metal")
			return MaterialType::Metal;
		if (str == "Dielectric")
			return MaterialType::Dielectric;

		APPLOG("Invalid material type");

		return MaterialType::Unknown;
	}

	void ParseEntity(EntityData& entityData, const Config& entity)
	{
		ParseShape(*entityData.shapeData, entity.GetSection("Shape"));
		ParseMaterial(*entityData.materialData, entity.GetSection("Material"));
	}

	void ParseShape(ShapeData& shapeData, const Config& shape)
	{
		shapeData.type = ParseShapeTypeFromString(shape.GetString("Type"));
		*shapeData.position = ParseVector(shape.GetArray("Position"));
		shapeData.radius = shape.GetFloat("Radius");
	}

	void ParseMaterial(MaterialData& materialData, const Config& material)
	{
		materialData.type = ParseMaterialTypeFromString(material.GetString("Type"));
		*materialData.color = ParseVector(material.GetArray("Color"));
		materialData.fuzziness = material.GetFloat("Fuzziness");
		materialData.refractiveIndex = material.GetFloat("RefractiveIndex");
	}
}