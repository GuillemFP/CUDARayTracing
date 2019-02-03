#ifndef MODULEINPUT_H
#define MODULEINPUT_H

#include "Module.h"
#include "Point.h"

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
	update_status PreUpdate(float dt);
	bool CleanUp();

	KeyState GetKey(int id) const { return _keyboard[id]; }
	KeyState GetMouseButtonDown(int id) const { return _mouseButtons[id - 1]; }
	bool GetWindowEvent(EventWindow code) const { return _bWindowEvents[code]; }
	const iPoint& GetMouseMotion() const { return _mouseMotion; }
	const iPoint& GetMousePosition() const { return _mousePosition; }
	const iPoint& GetMouseWheel() const { return _mouseWheel; }

private:
	iPoint _mouseMotion;
	iPoint _mousePosition;
	iPoint _mouseWheel;

	bool _bWindowEvents[WE_COUNT];
	KeyState* _keyboard;
	KeyState* _mouseButtons;
};

#endif // !MODULEINPUT_H