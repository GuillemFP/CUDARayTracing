#include "Application.h"
#include "Globals.h"

#include "SDL/include/SDL.h"
#pragma comment( lib, "SDL/libx64/SDL2.lib" )
#pragma comment( lib, "SDL/libx64/SDL2main.lib" )

enum main_states
{
	MAIN_CREATION,
	MAIN_START,
	MAIN_UPDATE,
	MAIN_FINISH,
	MAIN_EXIT
};

Application* App = nullptr;

int main(int argc, char ** argv)
{
	int main_return = EXIT_FAILURE;
	int update_return = NULL;
	main_states state = MAIN_CREATION;

	while (state != MAIN_EXIT)
	{
		switch (state)
		{
		case MAIN_CREATION:
			APPLOG("Application Creation ------------------");
			App = new Application();
			state = MAIN_START;
			break;
		case MAIN_START:
			APPLOG("Application Init ----------------------");
			if (App->Init() == false)
			{
				APPLOG("Application Init exits with error -----");
				state = MAIN_FINISH;
			}
			else
			{
				APPLOG("Application Update --------------------");
				state = MAIN_UPDATE;
			}
			break;
		case MAIN_UPDATE:
			update_return = App->Update();
			if (update_return != UPDATE_CONTINUE)
			{
				if (update_return == UPDATE_ERROR)
				{
					APPLOG("Application Update exits with error ---");
					state = MAIN_EXIT;
				}
				if (update_return == UPDATE_STOP)
					state = MAIN_FINISH;
			}
			break;
		case MAIN_FINISH:
			APPLOG("Application CleanUp -------------------");
			if (App->CleanUp() == false)
			{
				APPLOG("Application CleanUp exits with error --");
			}
			else
				main_return = EXIT_SUCCESS;
			state = MAIN_EXIT;
			break;
		}
	}

	return main_return;
}
