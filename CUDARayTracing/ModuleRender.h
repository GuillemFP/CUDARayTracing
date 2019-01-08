#ifndef MODULERENDER_H
#define MODULERENDER_H

#include "Module.h"

#define MODULERENDER_NAME "ModuleRender"

class SDL_Renderer;
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

	void DrawPixel(const Color& color, int x, int y);

private:
	SDL_Renderer* _renderer = nullptr;

	int _pixelsWidth = 0;
	int _pixelsHeight = 0;
};

#endif // !MODULERENDER_H