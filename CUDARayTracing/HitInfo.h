#ifndef HITINFO_H
#define HITINFO_H

#include "Vector3.h"

struct HitInfo
{
	bool isHit = false;
	float distance = 0.0f;
	Vector3 point;
	Vector3 normal;
};

#endif // !HITINFO_H
