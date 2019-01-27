#ifndef MODULERAYTRACING_H
#define MODULERAYTRACING_H

#define MODULERAYTRACING_NAME "ModuleRayTracing"

#include "Module.h"
#include "Color.h"
#include "Vector3.h"
#include <iostream>
#include <fstream>
#include <curand_kernel.h>

class Camera;
class Timer;
class EntityList;
class Screen;

class ModuleRayTracing : public Module
{
public:
	ModuleRayTracing();
	~ModuleRayTracing();

	bool Init(Config* config = nullptr);
	bool Start();
	bool CleanUp();

	update_status Update();

	int GetSamplesNumber() const;

private:
	void InitFile();
	void WriteColor(const Color& color);

private:
	bool _screenFinished = false;

	int _threadsX = 1;
	int _threadsY = 1;

	int _pixelsWidth = 0;
	int _pixelsHeight = 0;

	int _samplesPerPixel = 1;

	int _maxScatters = 10;
	float _minDistance = 0.0f;
	float _maxDistance = 1.0f;

	std::ofstream _ppmImage;

	EntityList** _entities;

	Camera* _camera = nullptr;
	Screen* _screen = nullptr;

	Timer* _rayTracingTime = nullptr;

    curandState* _dRandStates = nullptr;
};

#endif // !MODULERAYTRACING_H