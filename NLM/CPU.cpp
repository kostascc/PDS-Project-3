#include "CPU.h"

using namespace std;


namespace CPU
{

	int run(Parameters params)
	{

#ifdef DEBUG
		cout << "CPU Starting \n";
#endif

		// Start Clock
		utils::Clock clock = utils::Clock();
		clock.startClock();

		utils::ImageFile imgFile = utils::ImageFile();

		imgFile.Read(params.input.imgPath);

		imgFile.Write(params.input.outputDir + "/img2.bmp");
		

		cout << "size: " << imgFile.height << "x" << imgFile.width << "\n";

		return 0;

		cout << clock.stopClock() << "\n";

		return 0;
	}

}

