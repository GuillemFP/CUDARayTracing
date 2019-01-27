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

	checkCudaErrors(cudaMallocManaged((void**)&_colors, numberOfPixels * sizeof(Vector3)));
    //memset(_colors, 0.0f, 3.0f * numberOfPixels * sizeof(float));

	Config cameraConfig = config->GetSection("Camera");
	const Vector3 origin = ParseUtils::ParseVector(cameraConfig.GetArray("Position"));
	const Vector3 lookAt = ParseUtils::ParseVector(cameraConfig.GetArray("LookAt"));
	const Vector3 worldUp = ParseUtils::ParseVector(cameraConfig.GetArray("WorldUp"));
	const float fov = cameraConfig.GetFloat("Fov");

	_camera = new Camera(origin, lookAt, worldUp, fov, float(_pixelsWidth) / float(_pixelsHeight));

	checkCudaErrors(cudaMalloc((void **)&_entities, sizeof(EntityList)));
    RaytracingUtils::initEntities(_entities);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMalloc((void **)&_dRandStates, numberOfPixels * sizeof(curandState)));
    RaytracingUtils::initRender(_dRandStates, _pixelsWidth, _pixelsHeight, _threadsX, _threadsY);

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
	checkCudaErrors(cudaFree(_colors));

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
	if (_sampleCount >= _samplesPerPixel)
	{
		return UPDATE_CONTINUE;
	}

	_rayTracingTime->Start();

	RaytracingUtils::getColors(_colors, _entities, _camera, _dRandStates, _pixelsWidth, _pixelsHeight, _threadsX, _threadsY);

	const float seconds = _rayTracingTime->GetTimeInS();
	APPLOG("RayTracing sample finished after %f seconds", seconds);
	_rayTracingTime->Stop();

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

    ++_sampleCount;
	App->_renderer->DrawScreen(_colors, _sampleCount);

	return UPDATE_CONTINUE;
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

int ModuleRayTracing::GetInitialPixelY() const
{
	return _pixelsHeight - 1;
}

int ModuleRayTracing::GetInitialPixelX() const
{
	return 0;
}