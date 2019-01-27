#include "ModuleWindow.h"

#include "Config.h"
#include "SDL/include/SDL.h"
#include <stdio.h>

namespace
{
    const int kTitleLength = 200;
}

ModuleWindow::ModuleWindow() : Module(MODULEWINDOW_NAME)
{
}

ModuleWindow::~ModuleWindow()
{
}

bool ModuleWindow::Init(Config* config)
{
	_windowsWidth = config->GetIntRequired("WindowWidth");
	_windowsHeight = config->GetIntRequired("WindowHeight");

	APPLOG("Init SDL window & surface");

	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		APPLOG("SDL_VIDEO could not initialize! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	Uint32 flags = SDL_WINDOW_SHOWN;

	_window = SDL_CreateWindow("Ray Tracing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, _windowsWidth, _windowsHeight, flags);

	if (!_window)
	{
		APPLOG("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	_screenSurface = SDL_GetWindowSurface(_window);

    _windowsTitle = new char[kTitleLength];

	return true;
}

bool ModuleWindow::CleanUp()
{
    RELEASE_ARRAY(_windowsTitle);

	//Destroy window
	if (_window)
	{
		SDL_DestroyWindow(_window);
	}

	//Quit SDL subsystems
	SDL_Quit();
	return true;
}

void ModuleWindow::SetTitle(float fps, int numberOfSamples)
{
    int app_name_lenght = sprintf_s(_windowsTitle, kTitleLength, "");
    sprintf_s(_windowsTitle, kTitleLength, "FPS: %f, Number of samples: %d", fps, numberOfSamples);
    SDL_SetWindowTitle(_window, _windowsTitle);
}
