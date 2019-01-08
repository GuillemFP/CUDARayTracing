#ifndef MODULEINPUT_H
#define MODULEINPUT_H

#include "Module.h"

#define MODULEINPUT_NAME "ModuleInput"

enum KeyState
{
	KEY_IDLE = 0,
	KEY_DOWN,
	KEY_REPEAT,
	KEY_UP
};

enum EventWindow
{
	WE_QUIT = 0,
	WE_HIDE = 1,
	WE_SHOW = 2,
	WE_COUNT
};

class ModuleInput : public Module
{
public:
	ModuleInput();
	~ModuleInput();

	bool Init(Config* config = nullptr);
	update_status PreUpdate();
	bool CleanUp();

private:
	KeyState* _keyboard;
	KeyState* _mouseButtons;
};

#endif // !MODULEINPUT_H