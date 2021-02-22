#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <ctime>
#include <chrono>
#include <thread>

using namespace std;


namespace utils
{

	class Clock
	{
	private:
		clock_t clockT;

	public:
		Clock();
		void startClock();
		string stopClock();
		void sleep(int ms);
	};

}

#endif
