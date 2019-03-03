#include "ModuleRayTracing.h"

#include "Application.h"
#include "Camera.h"
#include "Color.h"
#include "Config.h"
#include "CudaUtils.h"
#include "EntityList.h"
#include "EntityData.h"
#include "Globals.h"
#include "Math.h"
#include "ModuleCamera.h"
#include "ModuleRender.h"
#include "ModuleWindow.h"
#include "ParseUtils.h"
#include "RaytracingUtils.h"
#include "Timer.h"
#include "ParseUtils.h"
#include "Screen.h"
#include <algorithm>

ModuleRayTracing::ModuleRayTracing() : Module(MODULERAYTRACING_NAME)
{
}

ModuleRayTracing::~ModuleRayTracing()
{
}

bool ModuleRayTracing::Init(Config* config)
{
	_rayTracingTime = new Timer();
	_rayTracingTime->Start();

	_resetTimer = new Timer();

	_minScatters = config->GetInt("MinScatters");
	_maxScatters = config->GetInt("MaxScatters");
	_currentScatters = _maxScatters;

	_minDistance = config->GetFloat("MinDistance");
	_maxDistance = config->GetFloat("MaxDistance");
	_waitingTime = config->GetFloat("WaitingTime");

	_samplesPerPixel = config->GetFloat("SamplesPerPixel");

	_threadsX = config->GetInt("ThreadsX", 1);
	_threadsY = config->GetInt("ThreadsY", 1);

	_pixelsWidth = App->_window->GetWindowsWidth();
	_pixelsHeight = App->_window->GetWindowsHeight();

	Config entities = Config(ENTITIES_CONFIGFILE);
	ConfigArray entitiesArray = entities.GetArray("Entities");
	ParseEntities(entitiesArray);

    const unsigned numberOfPixels = _pixelsWidth * _pixelsHeight;

	checkCudaErrors(cudaMalloc((void **)&_entities, sizeof(EntityList)));
    RaytracingUtils::initEntities(_entities, _entitiesData, _numEntities);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMalloc((void **)&_dRandStates, numberOfPixels * sizeof(curandState)));
    RaytracingUtils::initRender(_dRandStates, _pixelsWidth, _pixelsHeight, _threadsX, _threadsY);

	_screen = new Screen(_pixelsWidth, _pixelsHeight, _samplesPerPixel);
	_screen->Init();

	InitFile();

	const float seconds = _rayTracingTime->GetTimeInS();
	APPLOG("RayTracing initialiation end after %f seconds", seconds);
	_rayTracingTime->Stop();

	return true;
}

bool ModuleRayTracing::Start()
{
	return true;
}

bool ModuleRayTracing::CleanUp()
{
	checkCudaErrors(cudaFree(_screen));
	RELEASE_ARRAY(_entitiesData);
	//checkCudaErrors(cudaFree(_entitiesData));
	
	RaytracingUtils::cleanUpEntities(_entities);
	checkCudaErrors(cudaFree(_entities));

    checkCudaErrors(cudaFree(_dRandStates));

	_ppmImage.close();

	RELEASE(_resetTimer);
	RELEASE(_rayTracingTime);

	return true;
}

update_status ModuleRayTracing::Update(float dt)
{
	if (_resetTimer->IsRunning() && _resetTimer->GetTimeInS() >= _waitingTime)
	{
		_resetTimer->Stop();
		_currentScatters = _maxScatters;

		_screen->ResetScreen();
	}

	if (_screen->IsCompleted())
	{
		return UPDATE_CONTINUE;
	}

	_rayTracingTime->Start();

	RaytracingUtils::getColors(_screen, _entities, App->_camera->GetCamera(), _dRandStates, _pixelsWidth, _pixelsHeight, _threadsX, _threadsY, _currentScatters);

	const float seconds = _rayTracingTime->GetTimeInS();
	APPLOG("RayTracing sample finished after %f seconds", seconds);
	_rayTracingTime->Stop();

	App->_renderer->DrawScreen(_screen);

	return UPDATE_CONTINUE;
}

int ModuleRayTracing::GetSamplesNumber() const
{
	return _screen->GetSampleNumber();
}

void ModuleRayTracing::OnCameraMove()
{
	_screen->ResetScreen();

	_currentScatters = _minScatters;
	_resetTimer->Start();
}

void ModuleRayTracing::InitFile()
{
	_ppmImage.open("image.ppm");
	_ppmImage << "P3\n" << _pixelsWidth << " " << _pixelsHeight << "\n255\n";
}

void ModuleRayTracing::WriteColor(const Color& color)
{
	int ir = int(255.99*color.r);
	int ig = int(255.99*color.g);
	int ib = int(255.99*color.b);
	_ppmImage << ir << " " << ig << " " << ib << "\n";
}

void ModuleRayTracing::ParseEntities(const ConfigArray& entities)
{
	_numEntities = entities.GetArrayLength();
	_entitiesData = new EntityData[_numEntities];

	for (size_t i = 0; i < _numEntities; i++)
	{
		const Config entity = entities.GetSection(i);
		ParseUtils::ParseEntity(_entitiesData[i], entity);
	}
}
