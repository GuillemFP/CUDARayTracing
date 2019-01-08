#ifndef MODULEWINDOW_H
#define MODULEWINDOW_H

#include "Module.h"

#define MODULEWINDOW_NAME "ModuleWindow"

class SDL_Window;
class SDL_Surface;

class ModuleWindow : public Module
{
public:
	ModuleWindow();
	~ModuleWindow();

	bool Init(Config* config = nullptr);
	bool CleanUp();

	int GetWindowsWidth() const
	{
		return _windowsWidth;
	}

	int GetWindowsHeight() const
	{
		return _windowsHeight;
	}

	SDL_Window* GetWindow() const
	{
		return _window;
	}

private:
	SDL_Window* _window = nullptr;
	SDL_Surface* _screenSurface = nullptr;

	int _windowsWidth = 0;
	int _windowsHeight = 0;
};

#endif // !MODULEWINDOW_H