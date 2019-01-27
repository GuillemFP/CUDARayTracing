#include "ModuleRender.h"

#include "Application.h"
#include "Color.h"
#include "ModuleWindow.h"
#include "Vector3.h"
#include "Timer.h"
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

	_pixelsWidth = App->_window->GetWindowsWidth();
	_pixelsHeight = App->_window->GetWindowsHeight();

	_texture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, _pixelsWidth, _pixelsHeight);
	if (!_texture)
	{
		APPLOG("Texture could not be created! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	const size_t size = _pixelsWidth * _pixelsHeight;
	_pixels = new Uint32[size];
	memset(_pixels, 255, size * sizeof(Uint32));

	_timer = new Timer;

	return true;
}

bool ModuleRender::Start()
{
	return true;
}

bool ModuleRender::CleanUp()
{
	APPLOG("Destroying renderer");

	RELEASE(_timer);

	RELEASE_ARRAY(_pixels);

	if (_texture)
	{
		SDL_DestroyTexture(_texture);
	}

	//Destroy window
	if (_renderer)
	{
		SDL_DestroyRenderer(_renderer);
	}

	return true;
}

update_status ModuleRender::PostUpdate()
{
	SDL_RenderClear(_renderer);
	SDL_RenderCopy(_renderer, _texture, NULL, NULL);
	SDL_RenderPresent(_renderer);

	return UPDATE_CONTINUE;
}

void ModuleRender::DrawScreen(const Vector3* colors, int samples)
{
	_timer->Start();

	for (int j = 0; j < _pixelsHeight; j++) 
	{
		for (int i = 0; i < _pixelsWidth; i++)
		{
			const size_t index = j * _pixelsWidth + i;
			const int ir = int(255.99*colors[index].e[0] / samples);
			const int ig = int(255.99*colors[index].e[1] / samples);
			const int ib = int(255.99*colors[index].e[2] / samples);

			const size_t pixelIndex = (_pixelsHeight - 1 - j) * _pixelsWidth + i;
			const Uint32 color = (SDL_ALPHA_OPAQUE << 24) | ((ir & 0xFF) << 16) | ((ig & 0xFF) << 8) | (ib & 0xFF);
			_pixels[pixelIndex] = color;
		}
	}

	SDL_UpdateTexture(_texture, NULL, _pixels, _pixelsWidth * sizeof(Uint32));

	float seconds = _timer->GetTimeInS();
	APPLOG("Loading pixels finished after %f seconds", seconds);
	_timer->Stop();
}
