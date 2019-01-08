#include "Application.h"

#include "ModuleInput.h"
#include "ModuleRayTracing.h"
#include "ModuleRender.h"
#include "ModuleWindow.h"

#include "Config.h"

Application::Application()
{
	_modules.push_back(_input = new ModuleInput());
	_modules.push_back(_window = new ModuleWindow());
	_modules.push_back(_renderer = new ModuleRender());

	//_modules.push_back(_camera = new ModuleCamera());
	//_modules.push_back(_materials = new ModuleMaterials());
	//_modules.push_back(_entities = new ModuleEntities());
	_modules.push_back(_rayTracing = new ModuleRayTracing());
}

Application::~Application()
{
	for (Module* module : _modules)
	{
		RELEASE(module);
	}
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
	update_status ret = UPDATE_CONTINUE;

	for (std::vector<Module*>::iterator it = _modules.begin(); it != _modules.end() && ret == UPDATE_CONTINUE; ++it)
		if ((*it)->IsEnabled())
			ret = (*it)->PreUpdate();

	for (std::vector<Module*>::iterator it = _modules.begin(); it != _modules.end() && ret == UPDATE_CONTINUE; ++it)
		if ((*it)->IsEnabled())
			ret = (*it)->Update();

	for (std::vector<Module*>::iterator it = _modules.begin(); it != _modules.end() && ret == UPDATE_CONTINUE; ++it)
		if ((*it)->IsEnabled())
			ret = (*it)->PostUpdate();

	return ret;
}

bool Application::CleanUp()
{
	bool ret = true;

	for (std::vector<Module*>::reverse_iterator it = _modules.rbegin(); it != _modules.rend() && ret; ++it)
		ret = (*it)->CleanUp();

	return ret;
}
