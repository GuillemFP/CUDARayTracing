#include "ModuleInput.h"

#include "SDL/include/SDL.h"
#include <cstring>

#define KEYBOARD_MAX_KEYS 300
#define MOUSE_NUM_BUTTONS 5

ModuleInput::ModuleInput() : Module(MODULEINPUT_NAME)
{
	_keyboard = new KeyState[KEYBOARD_MAX_KEYS];
	memset(_keyboard, KEY_IDLE, KEYBOARD_MAX_KEYS * sizeof(KeyState));

	_mouseButtons = new KeyState[MOUSE_NUM_BUTTONS];
	memset(_mouseButtons, KEY_IDLE, MOUSE_NUM_BUTTONS * sizeof(KeyState));
}

ModuleInput::~ModuleInput()
{
	RELEASE_ARRAY(_keyboard);
	RELEASE_ARRAY(_mouseButtons);
}

bool ModuleInput::Init(Config* config)
{
	APPLOG("Init SDL input event system");
	SDL_Init(0);

	if (SDL_InitSubSystem(SDL_INIT_EVENTS) < 0)
	{
		APPLOG("SDL_EVENTS could not initialize! SDL_Error: %s\n", SDL_GetError());
		return false;
	}

	return true;
}

update_status ModuleInput::PreUpdate(float dt)
{
	static SDL_Event event;

	_mouseMotion = { 0, 0 };
	_mouseWheel = { 0, 0 };
	memset(_bWindowEvents, false, WE_COUNT * sizeof(bool));

	const Uint8* keys = SDL_GetKeyboardState(NULL);

	for (int i = 0; i < KEYBOARD_MAX_KEYS; ++i)
	{
		if (keys[i] == 1)
		{
			if (_keyboard[i] == KEY_IDLE)
				_keyboard[i] = KEY_DOWN;
			else
				_keyboard[i] = KEY_REPEAT;
		}
		else
		{
			if (_keyboard[i] == KEY_REPEAT || _keyboard[i] == KEY_DOWN)
				_keyboard[i] = KEY_UP;
			else
				_keyboard[i] = KEY_IDLE;
		}
	}

	for (int i = 0; i < MOUSE_NUM_BUTTONS; ++i)
	{
		if (_mouseButtons[i] == KEY_DOWN)
			_mouseButtons[i] = KEY_REPEAT;

		if (_mouseButtons[i] == KEY_UP)
			_mouseButtons[i] = KEY_IDLE;
	}

	while (SDL_PollEvent(&event) != 0)
	{
		switch (event.type)
		{
		case SDL_QUIT:
			_bWindowEvents[WE_QUIT] = true;
			break;
		case SDL_KEYDOWN:
		case SDL_KEYUP:
			//PrintKeyInfo(&event.key);
			break;
		case SDL_WINDOWEVENT:
			switch (event.window.event)
			{
			case SDL_WINDOWEVENT_HIDDEN:
			case SDL_WINDOWEVENT_MINIMIZED:
			case SDL_WINDOWEVENT_FOCUS_LOST:
				_bWindowEvents[WE_HIDE] = true;
				break;
			case SDL_WINDOWEVENT_SHOWN:
			case SDL_WINDOWEVENT_FOCUS_GAINED:
			case SDL_WINDOWEVENT_MAXIMIZED:
			case SDL_WINDOWEVENT_RESTORED:
				_bWindowEvents[WE_SHOW] = true;
				break;

			case SDL_WINDOWEVENT_SIZE_CHANGED:
				break;
			}

		case SDL_MOUSEBUTTONDOWN:
			_mouseButtons[event.button.button - 1] = KEY_DOWN;
			break;

		case SDL_MOUSEBUTTONUP:
			_mouseButtons[event.button.button - 1] = KEY_UP;
			break;

		case SDL_MOUSEMOTION:
			_mouseMotion.x = event.motion.xrel;
			_mouseMotion.y = event.motion.yrel;
			_mousePosition.x = event.motion.x;
			_mousePosition.y = event.motion.y;
			break;
		case SDL_MOUSEWHEEL:
			_mouseWheel.y = event.wheel.y;
			break;
		}
	}

	if (GetWindowEvent(EventWindow::WE_QUIT) == true || GetKey(SDL_SCANCODE_ESCAPE) == KEY_DOWN)
		return UPDATE_STOP;

	return UPDATE_CONTINUE;
}

bool ModuleInput::CleanUp()
{
	APPLOG("Quitting SDL event subsystem");
	SDL_QuitSubSystem(SDL_INIT_EVENTS);
	return true;
}
