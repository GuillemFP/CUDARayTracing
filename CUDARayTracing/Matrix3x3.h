#ifndef MATRIX3X3_H
#define MATRIX3X3_H

#include "Vector3.h"

class Matrix3x3
{
public:
	__host__ __device__ Matrix3x3() {}
	__host__ __device__ Matrix3x3(const Vector3& rotationAxis, const float rotationAngle)
	{
		const float cosA = cosf(rotationAngle);
		const float sinA = sinf(rotationAngle);

		const float mCosA = 1.0f - cosA;
		const float mSinA = 1.0f - sinA;

		const float x = rotationAxis.x();
		const float y = rotationAxis.y();
		const float z = rotationAxis.z();

		e[0][0] = cosA + x * x * mCosA; e[0][1] = x * y * mCosA - z * sinA; e[0][2] = x * z * mCosA + y * sinA;
		e[1][0] = y * x * mCosA + z * sinA; e[1][1] = cosA + y * y * mCosA; e[1][2] = y * z * mCosA - x * sinA;
		e[2][0] = z * x * mCosA - y * sinA; e[2][1] = z * y * mCosA + x * sinA; e[2][2] = cosA + z * z * mCosA;
	}

	__host__ __device__ inline Vector3 operator*(const Vector3& v) const
	{
		return Vector3(e[0][0] * v.x() + e[0][1] * v.y() + e[0][2] * v.z(), e[1][0] * v.x() + e[1][1] * v.y() + e[1][2] * v.z(), e[2][0] * v.x() + e[2][1] * v.y() + e[2][2] * v.z());
	}

	float e[3][3];
};

#endif // !MATRIX3X3_H
