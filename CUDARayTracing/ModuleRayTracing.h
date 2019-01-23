#ifndef MODULERAYTRACING_H
#define MODULERAYTRACING_H

#define MODULERAYTRACING_NAME "ModuleRayTracing"

#include "Module.h"
#include "Color.h"
#include "Vector3.h"
#include <iostream>
#include <fstream>

class Camera;
class Timer;
class Entity;

class ModuleRayTracing : public Module
{
public:
	ModuleRayTracing();
	~ModuleRayTracing();

	bool Init(Config* config = nullptr);
	bool Start();
	bool CleanUp();

	update_status Update();

private:
	void InitFile();
	void WriteColor(const Color& color);

	int GetInitialPixelY() const;
	int GetInitialPixelX() const;

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

	Entity** _list;
	Entity** _entities;

	Camera* _camera = nullptr;
	Vector3* _colors = nullptr;

	Timer* _rayTracingTime = nullptr;
};

#endif // !MODULERAYTRACING_H