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

update_status ModuleInput::PreUpdate()
{
	static SDL_Event event;

	if (SDL_PollEvent(&event) && event.type == SDL_QUIT)
	{
		return UPDATE_STOP;
	}

	return UPDATE_CONTINUE;
}

bool ModuleInput::CleanUp()
{
	APPLOG("Quitting SDL event subsystem");
	SDL_QuitSubSystem(SDL_INIT_EVENTS);
	return true;
}
