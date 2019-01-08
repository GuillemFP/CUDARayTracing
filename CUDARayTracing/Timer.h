#ifndef TIMER_H
#define TIMER_H

#include "SDL/include/SDL_timer.h"

class Timer
{
public:
	Timer() {};
	~Timer() {};

	void Start()
	{
		accumulated_ms = 0;
		start_time = SDL_GetTicks();
		running = true;
	}

	void Stop()
	{
		final_time = SDL_GetTicks();
		accumulated_ms = 0;
		running = false;
	}

	void Pause()
	{
		if (running)
		{
			final_time = SDL_GetTicks();
			accumulated_ms += final_time - start_time;
			running = false;
		}
	}

	void Unpause()
	{
		if (!running)
		{
			start_time = SDL_GetTicks();
			running = true;
		}
	}

	bool IsRunning() const { return running; }

	Uint32 GetTimeInMs() const
	{
		Uint32 ret;

		if (running)
			ret = SDL_GetTicks() - start_time + accumulated_ms;
		else
			ret = accumulated_ms;

		return ret;
	}

	float GetTimeInS() const
	{
		return GetTimeInMs() / 1000.0f;
	}

	void static DelayInMs(Uint32 delay)
	{
		SDL_Delay(delay);
	}

private:
	Uint32 start_time = 0;
	Uint32 final_time = 0;
	Uint32 accumulated_ms = 0;
	bool running = false;

};

#endif // !TIMER_H