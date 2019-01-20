#include "ModuleRayTracing.h"

#include "Application.h"
#include "Camera.h"
#include "Color.h"
#include "Config.h"
#include "CudaUtils.h"
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

	_maxScatters = config->GetInt("MaxScatters");
	_minDistance = config->GetFloat("MinDistance");
	_maxDistance = config->GetFloat("MaxDistance");

	_samplesPerPixel = config->GetFloat("SamplesPerPixel");

	_threadsX = config->GetInt("ThreadsX", 1);
	_threadsY = config->GetInt("ThreadsY", 1);

	_pixelsWidth = App->_window->GetWindowsWidth();
	_pixelsHeight = App->_window->GetWindowsHeight();

	size_t size = _pixelsWidth * _pixelsHeight * sizeof(Vector3);
	checkCudaErrors(cudaMallocManaged((void**)&_colors, size));

	Config cameraConfig = config->GetSection("Camera");
	const Vector3 origin = ParseUtils::ParseVector(cameraConfig.GetArray("Position"));
	const Vector3 lookAt = ParseUtils::ParseVector(cameraConfig.GetArray("LookAt"));
	const Vector3 worldUp = ParseUtils::ParseVector(cameraConfig.GetArray("WorldUp"));
	const float fov = cameraConfig.GetFloat("Fov");

	_camera = new Camera(origin, lookAt, worldUp, fov, float(_pixelsWidth) / float(_pixelsHeight));

	InitFile();

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

	_ppmImage.close();

	RELEASE(_rayTracingTime);

	return true;
}

update_status ModuleRayTracing::Update()
{
	if (_screenFinished)
	{
		return UPDATE_CONTINUE;
	}

	_rayTracingTime->Start();

	RaytracingUtils::getColors(_colors, _camera, _pixelsWidth, _pixelsHeight, _threadsX, _threadsY);

	float seconds = _rayTracingTime->GetTimeInS();
	APPLOG("RayTracing sample finished after %f seconds", seconds);
	_rayTracingTime->Stop();

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	App->_renderer->DrawScreen(_colors);

	_screenFinished = true;

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