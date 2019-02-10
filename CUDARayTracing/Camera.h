#ifndef CAMERA_H
#define CAMERA_H

#include "Managed.h"
#include "Vector3.h"
#include "Matrix3x3.h"
#include "Ray.h"
#include "RaytracingUtils.h"

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679f

class Camera : public Managed
{
public:
	__host__ __device__ Camera(const Vector3& position, const Vector3& lookAt, const Vector3& worldUp, float verticalFov, float aspectRatio, float aperture) : _position(position), _worldUp(worldUp), _verticalFov(verticalFov), _aspectRatio(aspectRatio), _lensRadius(0.5f * aperture)
	{
		LookAt(lookAt);
	}

	__host__ __device__ void LookAt(const Vector3& lookAt)
	{
		Vector3 toLookAt = _position - lookAt;
		_focusDistance = toLookAt.length();

		const float theta = _verticalFov * PI / 180.0f;
		_halfHeight = tanf(0.5f * theta);
		_halfWidth = _aspectRatio * _halfHeight;

		Vector3 w = normalize(toLookAt);
		_cameraFront = -w;
		_cameraRight = normalize(cross(_worldUp, w));
		_cameraUp = cross(w, _cameraRight);

		RecalculateViewport();
	}

	__device__ Ray GenerateRay(float widthFactor, float heightFactor, curandState* rand) const
	{
		Vector3 viewportPosition = _cornerBottomLeft + _viewportWidthVector * widthFactor + _viewportHeightVector * heightFactor;
		
		Vector3 randomInDisk = _lensRadius * MathUtils::RandomPointInDisk(rand);
		Vector3 rayOrigin = _position + _cameraRight * randomInDisk.x() + _cameraUp * randomInDisk.y();

		Vector3 unitVector = normalize(viewportPosition - rayOrigin);
		return Ray(rayOrigin, unitVector, 0);
	}

	__host__ __device__ const Vector3& GetFront() const { return _cameraFront; }
	__host__ __device__ const Vector3& GetRight() const { return _cameraRight; }
	__host__ __device__ const Vector3& GetUp() const { return _cameraUp; }

	__host__ __device__ float GetFocusDistance() const { return _focusDistance; }

	__host__ __device__ void Translate(const Vector3& translate)
	{
		_position += translate;

		RecalculateViewport();
	}

	__host__ __device__ void Rotate(const float xAngle, const float yAngle)
	{
		Matrix3x3 yRotation = Matrix3x3(_worldUp, xAngle);
		Matrix3x3 xRotation = Matrix3x3(_cameraRight, yAngle);
		Matrix3x3 rotation = yRotation * xRotation;

		_cameraFront = rotation * _cameraFront;
		_cameraUp = rotation * _cameraUp;
		_cameraRight = rotation * _cameraRight;

		RecalculateViewport();
	}

	__host__ __device__ void ChangeFocusDistance(const float dr)
	{
		_focusDistance += dr;

		if (_focusDistance < 1.0f)
		{
			_focusDistance = 1.0f;
		}

		RecalculateViewport();
	}

	__host__ __device__ void RecalculateViewport()
	{
		_viewportWidthVector = 2 * _halfWidth * _focusDistance * _cameraRight;
		_viewportHeightVector = 2 * _halfHeight * _focusDistance * _cameraUp;
		_cornerBottomLeft = _position - 0.5f * _viewportWidthVector - 0.5f * _viewportHeightVector + _cameraFront * _focusDistance;
	}

private:
	Vector3 _position;
	Vector3 _worldUp;

	Vector3 _cameraFront;
	Vector3 _cameraUp;
	Vector3 _cameraRight;

	Vector3 _cornerBottomLeft;
	Vector3 _viewportWidthVector;
	Vector3 _viewportHeightVector;

	float _verticalFov;
	float _aspectRatio;
	float _halfWidth = 1.0f;
	float _halfHeight = 1.0f;;

	float _lensRadius = 0.0f;
	float _focusDistance = 1.0f;

	float _minTime = 0.0f;
	float _maxTime = 0.0f;
};

#endif // !CAMERA_H