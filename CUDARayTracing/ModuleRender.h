#ifndef MODULERENDER_H
#define MODULERENDER_H

#include "Module.h"
#include "SDL/include/SDL_stdinc.h"

#define MODULERENDER_NAME "ModuleRender"

class SDL_Renderer;
class SDL_Texture;
class Vector3;
class Timer;
struct Color;

class ModuleRender : public Module
{
public:
	ModuleRender();
	~ModuleRender();

	bool Init(Config* config = nullptr);
	bool Start();
	bool CleanUp();

	update_status PostUpdate();

	void DrawScreen(const Vector3* colors);

private:
	SDL_Renderer* _renderer = nullptr;
	SDL_Texture* _texture = nullptr;
	Timer* _timer = nullptr;

	Uint32* _pixels = nullptr;
	int _pixelsWidth = 0;
	int _pixelsHeight = 0;
};

#endif // !MODULERENDER_H