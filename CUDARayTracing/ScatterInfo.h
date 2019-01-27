#ifndef SCATTERINFO_H
#define SCATTERINFO_H

#include "Ray.h"
#include "Vector3.h"

struct ScatterInfo
{
	bool scatters = false;
	Vector3 attenuation;
	Ray scatteredRay;
};

#endif // !SCATTERINFO_H
