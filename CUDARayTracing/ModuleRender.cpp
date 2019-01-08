#include "ModuleRender.h"

#include "Application.h"
#include "Color.h"
#include "ModuleWindow.h"
#include "SDL/include/SDL.h"

ModuleRender::ModuleRender() : Module(MODULERENDER_NAME)
{
}

ModuleRender::~ModuleRender()
{
}

bool ModuleRender::Init(Config* config)
{
	APPLOG("Creating Renderer context");

	Uint32 flags = 0;
	_renderer = SDL_CreateRenderer(App->_window->GetWindow(), -1, flags);

	if (!_renderer)
	{
		APPLOG("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	SDL_SetRenderDrawColor(_renderer, 0, 0, 0, 0);
	SDL_RenderClear(_renderer);

	_pixelsWidth = App->_window->GetWindowsWidth();
	_pixelsHeight = App->_window->GetWindowsHeight();

	return true;
}

bool ModuleRender::Start()
{
	return true;
}

bool ModuleRender::CleanUp()
{
	APPLOG("Destroying renderer");

	//Destroy window
	if (_renderer)
	{
		SDL_DestroyRenderer(_renderer);
	}

	return true;
}

update_status ModuleRender::PostUpdate()
{
	SDL_RenderPresent(_renderer);

	return UPDATE_CONTINUE;
}

void ModuleRender::DrawPixel(const Color& color, int x, int y)
{
	int ir = int(255.99*color.r);
	int ig = int(255.99*color.g);
	int ib = int(255.99*color.b);
	SDL_SetRenderDrawColor(_renderer, ir, ig, ib, 255);
	SDL_RenderDrawPoint(_renderer, x, _pixelsHeight - y);
}