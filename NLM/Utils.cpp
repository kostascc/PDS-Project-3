#include "Utils.h"


utils::Clock::Clock() {
	clockT = clock_t();
}


void utils::Clock::startClock()
{
	clockT = clock();
	return;
}


string utils::Clock::stopClock()
{
	char buf[100];
	snprintf(buf, sizeof(buf), "%.4fs", (double)(clock() - clockT) / CLOCKS_PER_SEC);
	string bufStr = buf;
	return bufStr;
}

void utils::Clock::sleep(int ms)
{
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
	return;
}



utils::ImageFile::ImageFile()
{
	;
}



/**
 * Written by Elejandro Rodriguez
 **/
void utils::ImageFile::Read(string filePath)
{

	//Open the file for reading in binary mode
	FILE* imageFile = fopen(filePath.c_str(), "rb");
	//Read data offset
	int32 dataOffset;
	fseek(imageFile, DATA_OFFSET_OFFSET, SEEK_SET);
	fread(&dataOffset, 4, 1, imageFile);
	//Read width
	fseek(imageFile, WIDTH_OFFSET, SEEK_SET);
	fread(&width, 4, 1, imageFile);
	//Read height
	fseek(imageFile, HEIGHT_OFFSET, SEEK_SET);
	fread(&height, 4, 1, imageFile);
	//Read bits per pixel
	int16 bitsPerPixel;
	fseek(imageFile, BITS_PER_PIXEL_OFFSET, SEEK_SET);
	fread(&bitsPerPixel, 2, 1, imageFile);
	//Allocate a pixel array
	bytesPerPixel = ((int32)bitsPerPixel) / 8;

	cout << "Read: " << height << "x" << width << ", bytesPerPixel:" << bytesPerPixel << "\n";

	//Rows are stored bottom-up
	//Each row is padded to be a multiple of 4 bytes. 
	//We calculate the padded row size in bytes
	int paddedRowSize = (int)(4 * ceil((float)(width) / 4.0f)) * (bytesPerPixel);
	//We are not interested in the padded bytes, so we allocate memory just for
	//the pixel data
	int unpaddedRowSize = (width) * (bytesPerPixel);
	//Total size of the pixel data in bytes
	int totalSize = unpaddedRowSize * (height);
	pixelArr = (byte*)malloc(totalSize);
	//Read the pixel data Row by Row.
	//Data is padded and stored bottom-up
	int i = 0;
	//point to the last row of our pixel array (unpadded)
	byte* currentRowPointer = pixelArr + ((height - 1) * unpaddedRowSize);
	for (i = 0; i < height; i++)
	{
		//put file cursor in the next row from top to bottom
		fseek(imageFile, dataOffset + (i * paddedRowSize), SEEK_SET);
		//read only unpaddedRowSize bytes (we can ignore the padding bytes)
		fread(currentRowPointer, 1, unpaddedRowSize, imageFile);
		//point to the next row (from bottom to top)
		currentRowPointer -= unpaddedRowSize;
	}

	fclose(imageFile);

	return;
}



/**
 * Written by Elejandro Rodriguez
 **/
void utils::ImageFile::Write(string filePath)
{
	//Open file in binary mode
	FILE* outputFile = fopen(filePath.c_str(), "wb");
	//*****HEADER************//
	//write signature
	const char* BM = "BM";
	fwrite(&BM[0], 1, 1, outputFile);
	fwrite(&BM[1], 1, 1, outputFile);
	//Write file size considering padded bytes
	int paddedRowSize = (int)(4 * ceil((float)width / 4.0f)) * bytesPerPixel;
	int32 fileSize = paddedRowSize * height + HEADER_SIZE + INFO_HEADER_SIZE;
	fwrite(&fileSize, 4, 1, outputFile);
	//Write reserved
	int32 reserved = 0x0000;
	fwrite(&reserved, 4, 1, outputFile);
	//Write data offset
	int32 dataOffset = HEADER_SIZE + INFO_HEADER_SIZE;
	fwrite(&dataOffset, 4, 1, outputFile);

	//*******INFO*HEADER******//
	//Write size
	int32 infoHeaderSize = INFO_HEADER_SIZE;
	fwrite(&infoHeaderSize, 4, 1, outputFile);
	//Write width and height
	fwrite(&width, 4, 1, outputFile);
	fwrite(&height, 4, 1, outputFile);
	//Write planes
	int16 planes = 1; //always 1
	fwrite(&planes, 2, 1, outputFile);
	//write bits per pixel
	int16 bitsPerPixel = bytesPerPixel * 8;
	fwrite(&bitsPerPixel, 2, 1, outputFile);
	//write compression
	int32 compression = NO_COMPRESION;
	fwrite(&compression, 4, 1, outputFile);
	//write image size (in bytes)
	int32 imageSize = width * height * bytesPerPixel;
	fwrite(&imageSize, 4, 1, outputFile);
	//write resolution (in pixels per meter)
	int32 resolutionX = 11811; //300 dpi
	int32 resolutionY = 11811; //300 dpi
	fwrite(&resolutionX, 4, 1, outputFile);
	fwrite(&resolutionY, 4, 1, outputFile);
	//write colors used 
	int32 colorsUsed = MAX_NUMBER_OF_COLORS;
	fwrite(&colorsUsed, 4, 1, outputFile);
	//Write important colors
	int32 importantColors = ALL_COLORS_REQUIRED;
	fwrite(&importantColors, 4, 1, outputFile);
	//write data
	int i = 0;
	int unpaddedRowSize = width * bytesPerPixel;
	for (i = 0; i < height; i++)
	{
		//start writing from the beginning of last row in the pixel array
		int pixelOffset = ((height - i) - 1) * unpaddedRowSize;
		fwrite(&pixelArr[pixelOffset], 1, paddedRowSize, outputFile);
	}
	fclose(outputFile);

	return;
}