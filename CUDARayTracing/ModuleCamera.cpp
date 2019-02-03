#include "ModuleCamera.h"

#include "Application.h"
#include "Camera.h"
#include "Config.h"
#include "ParseUtils.h"
#include "ModuleWindow.h"
#include "ModuleInput.h"
#include "ModuleRayTracing.h"
#include "SDL/include/SDL.h"

ModuleCamera::ModuleCamera() : Module(MODULECAMERA_NAME)
{
}

ModuleCamera::~ModuleCamera()
{
}

bool ModuleCamera::Init(Config* config)
{
	Config cameraConfig = config->GetSection("Camera");
	const Vector3 origin = ParseUtils::ParseVector(cameraConfig.GetArray("Position"));
	const Vector3 lookAt = ParseUtils::ParseVector(cameraConfig.GetArray("LookAt"));
	_worldUp = ParseUtils::ParseVector(cameraConfig.GetArray("WorldUp"));
	const float fov = cameraConfig.GetFloat("Fov");

	int pixelsWidth = App->_window->GetWindowsWidth();
	int pixelsHeight = App->_window->GetWindowsHeight();
	_camera = new Camera(origin, lookAt, _worldUp, fov, float(pixelsWidth) / float(pixelsHeight));

	return true;
}

bool ModuleCamera::CleanUp()
{
	RELEASE(_camera);

	return true;
}

update_status ModuleCamera::Update(float dt)
{
	Vector3 movement = Vector3(0.0f, 0.0f, 0.0f);
	bool isMoving = false;

	if (App->_input->GetKey(SDL_SCANCODE_W) == KEY_REPEAT)
	{
		movement += _camera->GetFront();
		isMoving = true;
	}
	if (App->_input->GetKey(SDL_SCANCODE_S) == KEY_REPEAT)
	{
		movement -= _camera->GetFront();
		isMoving = true;
	}
	if (App->_input->GetKey(SDL_SCANCODE_A) == KEY_REPEAT)
	{
		movement -= _camera->GetRight();
		isMoving = true;
	}
	if (App->_input->GetKey(SDL_SCANCODE_D) == KEY_REPEAT)
	{
		movement += _camera->GetRight();
		isMoving = true;
	}
	if (App->_input->GetKey(SDL_SCANCODE_Q) == KEY_REPEAT)
	{
		movement -= _worldUp;
		isMoving = true;
	}
	if (App->_input->GetKey(SDL_SCANCODE_E) == KEY_REPEAT)
	{
		movement += _worldUp;
		isMoving = true;
	}

	if (isMoving)
	{
		App->_rayTracing->ResetImage();
		
		movement = 1.0f * dt * movement;
		_camera->Translate(movement);
	}

	return UPDATE_CONTINUE;
}

