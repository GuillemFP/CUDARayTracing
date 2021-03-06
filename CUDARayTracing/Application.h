#ifndef APPLICATION_H
#define APPLICATION_H

#include <vector>
#include "Globals.h"
#include "Module.h"

class ModuleInput;
class ModuleWindow;
class ModuleRender;

class ModuleCamera;
class ModuleMaterials;
class ModuleEntities;
class ModuleRayTracing;

class Timer;

class Application
{
public:
	Application();
	~Application();

	bool Init();
	update_status Update();
	bool CleanUp();

public:
	ModuleInput* _input;
	ModuleWindow* _window;
	ModuleRender* _renderer;
	
	ModuleCamera* _camera;
	ModuleMaterials* _materials;
	ModuleEntities* _entities;
	ModuleRayTracing* _rayTracing;

private:
	std::vector<Module*> _modules;

	Timer* _updateTimer = nullptr;

	int _frameCount = 0;
	int _framesLastS = 0;

	int _accumTimeMs = 0;
	int _prevTime = 0;
};

extern Application* App;

#endif // !APPLICATION_H