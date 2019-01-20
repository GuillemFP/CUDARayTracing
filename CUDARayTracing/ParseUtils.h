#ifndef PARSEUTILS_H
#define PARSEUTILS_H

#include "Vector3.h"

class Config;
class ConfigArray;

namespace ParseUtils
{
	Vector3 ParseVector(const ConfigArray& config, const Vector3& defaultValue = Vector3::one);
}

#endif // !PARSEUTILS_H