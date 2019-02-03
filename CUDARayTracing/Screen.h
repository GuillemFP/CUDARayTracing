#ifndef SCREEN_H
#define SCREEN_H

#include "Managed.h"
#include "Vector3.h"

class Screen : public Managed
{
public:
	__host__ __device__ Screen(int width, int height, int maxSamples) : _width(width), _height(height), _maxSamples(maxSamples), _samples(1)
	{
		checkCudaErrors(cudaMallocManaged((void**)&_accumColors, _width * _height * sizeof(Vector3)));
		checkCudaErrors(cudaMallocManaged((void**)&_pixels, _width * _height * sizeof(unsigned int)));

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__host__ __device__ ~Screen() 
	{
		checkCudaErrors(cudaFree(_pixels));
		checkCudaErrors(cudaFree(_accumColors));

		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__host__ __device__ bool Init()
	{
		ResetScreen();

		return true;
	}

	__host__ __device__ void AddSample()
	{
		if (!IsCompleted())
		{
			++_samples;
		}
	}

	__host__ __device__ bool IsCompleted()
	{
		return _samples >= _maxSamples;
	}

	__host__ __device__ int GetSampleNumber() const { return _samples; }

	__host__ __device__ unsigned int* GetPixels() const { return _pixels; }

	__host__ __device__ bool AddColor(const Vector3& color, int i, int j)
	{
		int index = (_height - 1 - j) * _width + i;

		_accumColors[index] += color;

		const int ir = int(255.99*_accumColors[index].e[0] / _samples);
		const int ig = int(255.99*_accumColors[index].e[1] / _samples);
		const int ib = int(255.99*_accumColors[index].e[2] / _samples);

		_pixels[index] = (255 << 24) | ((ir & 0xFF) << 16) | ((ig & 0xFF) << 8) | (ib & 0xFF);
	}

	__host__ __device__ void ResetScreen()
	{
		memset(_pixels, 0, _width * _height * sizeof(unsigned int));
		memset(_accumColors, 0.0f, _width * _height * sizeof(Vector3));

		_samples = 1;
	}

private:
	int _width, _height;
	int _maxSamples;
	int _samples = 1;

	Vector3* _accumColors = nullptr;
	unsigned int* _pixels = nullptr;
};

#endif // !SCREEN_H
