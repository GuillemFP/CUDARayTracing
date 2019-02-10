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
	float GetFocusDistance() const;

private:
	Camera* _camera = nullptr;

	Vector3 _worldUp;

	float _translationSpeed = 1.0f;
	float _rotationSpeed = 1.0f;
};

#endif // !MODULECAMERA_H
