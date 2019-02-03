#ifndef MODULECAMERA_H
#define MODULECAMERA_H

#include "Module.h"
#include "Vector3.h"

#define MODULECAMERA_NAME "ModuleCamera"

class Camera;

class ModuleCamera : public Module
{
public:
	ModuleCamera();
	~ModuleCamera();

	bool Init(Config* config);
	update_status Update(float dt);
	bool CleanUp();

	Camera* GetCamera() { return _camera; }

private:
	Camera* _camera = nullptr;

	Vector3 _worldUp;
};

#endif // !MODULECAMERA_H
