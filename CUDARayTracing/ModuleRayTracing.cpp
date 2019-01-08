#include "ModuleRayTracing.h"

#include "Application.h"
#include "Color.h"
#include "Config.h"
#include "CudaUtils.h"
#include "Globals.h"
#include "Math.h"
#include "ModuleRender.h"
#include "ModuleWindow.h"
#include <algorithm>

ModuleRayTracing::ModuleRayTracing() : Module(MODULERAYTRACING_NAME)
{
}

ModuleRayTracing::~ModuleRayTracing()
{
}

bool ModuleRayTracing::Init(Config* config)
{
	_maxScatters = config->GetInt("MaxScatters");
	_minDistance = config->GetFloat("MinDistance");
	_maxDistance = config->GetFloat("MaxDistance");

	_samplesPerPixel = config->GetFloat("SamplesPerPixel");

	_threadsX = config->GetInt("ThreadsX", 1);
	_threadsY = config->GetInt("ThreadsY", 1);

	_pixelsWidth = App->_window->GetWindowsWidth();
	_pixelsHeight = App->_window->GetWindowsHeight();
	_colorRow = new Color[_pixelsWidth];

	_currentY = GetInitialPixelY();

	size_t size = _pixelsWidth * _pixelsHeight * sizeof(Vector3);
	checkCudaErrors(cudaMallocManaged(&_colors, size));

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

	_ppmImage.close();

	RELEASE_ARRAY(_colorRow);

	return true;
}

update_status ModuleRayTracing::Update()
{
	if (_screenFinished)
	{
		return UPDATE_CONTINUE;
	}

	CudaUtils::getColors(_colors, _pixelsWidth, _pixelsHeight, _threadsX, _threadsY);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	for (int j = _pixelsHeight - 1; j >= 0; j--) {
		for (int i = 0; i < _pixelsWidth; i++) {
			size_t pixel_index = j * _pixelsWidth + i;
			App->_renderer->DrawPixel(_colors[pixel_index], i, j);
			WriteColor(_colors[pixel_index]);
		}
	}

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