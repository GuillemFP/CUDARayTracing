#include "ParseUtils.h"

#include "Config.h"

namespace ParseUtils
{
	Vector3 ParseVector(const ConfigArray& config, const Vector3& defaultValue)
	{
		return Vector3(config.GetFloat(0, defaultValue.x()), config.GetFloat(1, defaultValue.y()), config.GetFloat(2, defaultValue.z()));
	}
}