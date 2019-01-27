#include "ModuleRayTracing.h"

#include "Application.h"
#include "Camera.h"
#include "Color.h"
#include "Config.h"
#include "CudaUtils.h"
#include "EntityList.h"
#include "Globals.h"
#include "Math.h"
#include "ModuleRender.h"
#include "ModuleWindow.h"
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

	_maxScatters = config->GetInt("MaxScatters");
	_minDistance = config->GetFloat("MinDistance");
	_maxDistance = config->GetFloat("MaxDistance");

	_samplesPerPixel = config->GetFloat("SamplesPerPixel");

	_threadsX = config->GetInt("ThreadsX", 1);
	_threadsY = config->GetInt("ThreadsY", 1);

	_pixelsWidth = App->_window->GetWindowsWidth();
	_pixelsHeight = App->_window->GetWindowsHeight();

    const unsigned numberOfPixels = _pixelsWidth * _pixelsHeight;

	Config cameraConfig = config->GetSection("Camera");
	const Vector3 origin = ParseUtils::ParseVector(cameraConfig.GetArray("Position"));
	const Vector3 lookAt = ParseUtils::ParseVector(cameraConfig.GetArray("LookAt"));
	const Vector3 worldUp = ParseUtils::ParseVector(cameraConfig.GetArray("WorldUp"));
	const float fov = cameraConfig.GetFloat("Fov");

	checkCudaErrors(cudaMalloc((void **)&_entities, sizeof(EntityList)));
    RaytracingUtils::initEntities(_entities);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMalloc((void **)&_dRandStates, numberOfPixels * sizeof(curandState)));
    RaytracingUtils::initRender(_dRandStates, _pixelsWidth, _pixelsHeight, _threadsX, _threadsY);

	_camera = new Camera(origin, lookAt, worldUp, fov, float(_pixelsWidth) / float(_pixelsHeight));

	_screen = new Screen(_pixelsWidth, _pixelsHeight, _samplesPerPixel);

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

	RELEASE(_camera);
	
	RaytracingUtils::cleanUpEntities(_entities);
	checkCudaErrors(cudaFree(_entities));

    checkCudaErrors(cudaFree(_dRandStates));

	_ppmImage.close();

	RELEASE(_rayTracingTime);

	return true;
}

update_status ModuleRayTracing::Update()
{
	if (_screen->IsCompleted())
	{
		return UPDATE_CONTINUE;
	}

	_rayTracingTime->Start();

	RaytracingUtils::getColors(_screen, _entities, _camera, _dRandStates, _pixelsWidth, _pixelsHeight, _threadsX, _threadsY);

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