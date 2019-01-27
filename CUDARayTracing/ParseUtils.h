#ifndef PARSEUTILS_H
#define PARSEUTILS_H

#include "Vector3.h"

class Config;
class ConfigArray;

namespace ParseUtils
{
	Vector3 ParseVector(const ConfigArray& config, const Vector3& defaultValue = Vector3(1.0f, 1.0f, 1.0f));
}

#endif // !PARSEUTILS_H