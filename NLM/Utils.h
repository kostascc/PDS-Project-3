/**
 * (C) 2021 Konstantinos Chatzis
 * Aristotle University of Thessaloniki
 **/

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
#include <vector>


/*************************************
 *      Values for Bitmap I/O
 *************************************/
#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0
#define ONLY_8_BIT_BMP
#define ONLY_RECTANGULAR
#define NORMALIZE_PIXELS
#define FIX_PIXELS_OUT_OF_BOUND


#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif


typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;


using namespace std;


namespace utils
{

	/**************************
	 *      Program Timer
	 **************************/
	class Clock
	{
	private:

		clock_t clockT;		// Timer

	public:
		Clock();

		/**
		 * Record starting time
		 **/
		void startClock();

		/**
		 * Record Stop time and return the value in seconds
		 **/
		string stopClock();

		/**
		 * Pause Thread for milliseconds
		 **/
		void sleep(int ms);
	};


	/**************************
	 *  File / Bitmap Manager
	 **************************/
	class ImageFile
	{
	private:
		int mSize;			// Malloc Size

	public:
		
		float* pixelArr;	// Array of pixel values
		int height;			// Pixel Height
		int width;			// Pixel Width
		int bytesPerPixel;	// Bytes to use per pixel (when writing)

		ImageFile();

		/**
		 * Read BMP image file into 'pixelArr'.
		 **/
		void Read(string filePath);

		/**
		 * Write BMP image from 'pixelArr' into the file system.
		 **/
		void Write(string filePath);

		/**
		 * Get Filename part of path
		 **/
		static string GetFileName(string filePath);

	};


	/**************************
	 *      Miscellaneous
	 **************************/
	static class Func
	{
	public:

		/**
		 * Break String into a vector, by a delimiter
		 **/
		static vector<string> ExplodeString(string str, char del);
	};

}

#endif
