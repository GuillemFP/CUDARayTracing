#include "Application.h"

#include "ModuleInput.h"
#include "ModuleRayTracing.h"
#include "ModuleRender.h"
#include "ModuleWindow.h"
#include "ModuleCamera.h"

#include "Config.h"
#include "Timer.h"

Application::Application()
{
	_modules.push_back(_input = new ModuleInput());
	_modules.push_back(_window = new ModuleWindow());
	_modules.push_back(_renderer = new ModuleRender());

	_modules.push_back(_camera = new ModuleCamera());
	//_modules.push_back(_materials = new ModuleMaterials());
	//_modules.push_back(_entities = new ModuleEntities());
	_modules.push_back(_rayTracing = new ModuleRayTracing());

	_updateTimer = new Timer();
}

Application::~Application()
{
	for (Module* module : _modules)
	{
		RELEASE(module);
	}

	RELEASE(_updateTimer);
}

bool Application::Init()
{
	bool ret = true;

	Config config = Config(CONFIGFILE).GetSection("Config");

	for (std::vector<Module*>::iterator it = _modules.begin(); it != _modules.end() && ret; ++it)
	{
		const char* name = (*it)->GetName();
		Config section = config.GetSection(name);
		ret = (*it)->Init(&section);
	}
		//ret = (*it)->Init(&(config.GetSection((*it)->GetName())));

	for (std::vector<Module*>::iterator it = _modules.begin(); it != _modules.end() && ret; ++it)
	{
		if ((*it)->IsEnabled())
			ret = (*it)->Start();
	}

	return ret;
}

update_status Application::Update()
{
	float dt = _updateTimer->GetTimeInS();
	_updateTimer->Start();

	update_status ret = UPDATE_CONTINUE;

	for (std::vector<Module*>::iterator it = _modules.begin(); it != _modules.end() && ret == UPDATE_CONTINUE; ++it)
		if ((*it)->IsEnabled())
			ret = (*it)->PreUpdate(dt);

	for (std::vector<Module*>::iterator it = _modules.begin(); it != _modules.end() && ret == UPDATE_CONTINUE; ++it)
		if ((*it)->IsEnabled())
			ret = (*it)->Update(dt);

	for (std::vector<Module*>::iterator it = _modules.begin(); it != _modules.end() && ret == UPDATE_CONTINUE; ++it)
		if ((*it)->IsEnabled())
			ret = (*it)->PostUpdate(dt);

	++_frameCount;

	const Uint32 updateMs = _updateTimer->GetTimeInMs();
	_accumTimeMs += updateMs;

	if (_accumTimeMs > 1000)
	{
		_framesLastS = _frameCount;
		_frameCount = 0;
		_accumTimeMs -= 1000;
	}

	_window->SetTitle(_framesLastS, _rayTracing->GetSamplesNumber());

	return ret;
}

bool Application::CleanUp()
{
	bool ret = true;

	for (std::vector<Module*>::reverse_iterator it = _modules.rbegin(); it != _modules.rend() && ret; ++it)
		ret = (*it)->CleanUp();

	return ret;
}
