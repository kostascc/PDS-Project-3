#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <ctime>
#include <chrono>
#include <thread>
#include "Parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;


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


	class ImageFile
	{
	public:
		ImageFile();
		byte* pixelArr;
		int height;
		int width;
		int bytesPerPixel;
		void Read(string filePath);
		void Write(string filePath);

	};

}

#endif
