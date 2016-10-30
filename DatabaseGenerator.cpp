#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <dirent.h> 
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>

using namespace cv;
using namespace std;

//g++ DatabaseGenerator.cpp `pkg-config --libs opencv` -o databaseGenerator.out


// Fucntion declaration
void initDatabaseFolder();
void performSystemCommand(String command);
void splitByRegion();
void readAndRegionSplit(String imagePath);
void readAndPixelSplit(String imagePath);
void splitByPixel();
void robotPreProcessing();
void readAndPostProcess(String imagePath);

// Necessary paths 
String splitByRegionPath = "EmptyImages/SplitByRegion/";
String splitByPixelPath = "EmptyImages/SplitByPixel/";
String robotImagesPath = "RobotImages/";
String outputDatabasePath= "NewDatabase/";

// Global variables
int outputWidth = 64;
int outputHeight = 64;
Size outputSize = Size(outputWidth, outputHeight);
int emptyImageNr = 0;
int robotImageNr = 0;
int robotImageInputCount = 0;

// Step sized used by splitByPixel()
int stepsSize = 10;

// Variables for pre prosessing robot images
int rotDegrees[1] = {0};
double scales[3] = {1.0, 0.9, 0.8};
int guasianFilterSizes[6] = {1, 3, 5, 7, 9, 11};



int main(int argc, char* argv[])
{
	initDatabaseFolder();
	splitByRegion(); //Splits each image into a grid where each cell has the size 64x64
	splitByPixel(); //Slides a window (64x64) over the image and saves each window as an empty image 
	robotPreProcessing();
}

void initDatabaseFolder()
{
	struct stat info;
	if(stat( outputDatabasePath.c_str(), &info  ) == 0) // If outputDatabasePath does exist
	{
		// Delete outputDatabasePath
		performSystemCommand("exec rm -r " + outputDatabasePath);
	}

	// Create new outputDatabasePath directory
	performSystemCommand("exec mkdir " + outputDatabasePath);


	// Create folder for validation and testing
	performSystemCommand("exec mkdir " + outputDatabasePath + "Validation " + outputDatabasePath + "Training");

	// Create subfolders for empty and robot images
	performSystemCommand("exec mkdir " + outputDatabasePath + "Validation/empty " + outputDatabasePath + "Validation/robot");
	performSystemCommand("exec mkdir " + outputDatabasePath + "Training/empty " + outputDatabasePath + "Training/robot");
}

void performSystemCommand(String command)
{
    system(command.c_str());
}

void splitByRegion()
{
	// Finds all images in the splitByRegionPath directory and runs readAndRegionSplit for each
	DIR           *d;
	struct dirent *dir;
	String dirPath = splitByRegionPath + ".";
	d = opendir(dirPath.c_str());
	if (d)
	{
		while ((dir = readdir(d)) != NULL)
		{
			String imagePath = splitByRegionPath + dir->d_name;
			readAndRegionSplit(imagePath);
		}
	closedir(d);
	}  
}

void readAndRegionSplit(String imagePath)
{
	Mat image = cv::imread(imagePath);
	if (!image.empty())
	{
		Size s = image.size();
		int width = s.width;
		int height = s.height;
		int w = (width / outputWidth)-1;
		int h = (height / outputHeight)-1;
		for (int i=0; i < w*h; i++)
		{
			int x = (i%w) * outputWidth;
			int y = (i/w) * outputHeight;
			Rect myROI(x, y, outputWidth, outputHeight);
			Mat outputImage = image(myROI);

			//String outputFolder = emptyImageNr++%4 != 0 ? "Training/" : "Validation/";
			String outputFolder = "Training/";
			ostringstream outputName;
    		outputName << outputDatabasePath << outputFolder << "empty/" << emptyImageNr++ << ".jpg";
			imwrite(outputName.str(), outputImage);
		} 
	}
}


void splitByPixel()
{
	// Finds all images in the splitByPixelPath directory and runs readAndPixelSplit for each
	DIR           *d;
	struct dirent *dir;
	String dirPath = splitByPixelPath + ".";
	d = opendir(dirPath.c_str());
	if (d)
	{
		while ((dir = readdir(d)) != NULL)
		{
			String imagePath = splitByPixelPath + dir->d_name;
			readAndPixelSplit(imagePath);
		}
	closedir(d);
	}  
}

void readAndPixelSplit(String imagePath)
{
	Mat image = imread(imagePath); 
	if (!image.empty())
	{
		Size s = image.size();
		int width = s.width;
		int height = s.height;
		int x_steps = (width - outputWidth)/stepsSize;
    	int y_steps = (height - outputHeight)/stepsSize;

    	for(int y = 0; y < y_steps; y++)
    	{
    		for(int x = 0; x < x_steps; x++)
    		{
				Rect myROI(x*stepsSize, y*stepsSize, outputWidth, outputHeight);
				Mat outputImage = image(myROI);

				//String outputFolder = emptyImageNr++%4 != 0 ? "Training/" : "Validation/";
				String outputFolder = "Training/";
				ostringstream outputName;
    			outputName << outputDatabasePath << outputFolder << "empty/" << emptyImageNr++ << ".jpg";
				imwrite(outputName.str(), outputImage);
    		}
    	}
	}
}

void robotPreProcessing()
{
	DIR           *d;
	struct dirent *dir;
	String dirPath = robotImagesPath + ".";
	d = opendir(dirPath.c_str());
	if (d)
	{
		while ((dir = readdir(d)) != NULL)
		{
			String imagePath = robotImagesPath + dir->d_name;
			readAndPostProcess(imagePath);
		}
	closedir(d);
	}  
}

void readAndPostProcess(String imagePath)
{
	Mat img = cv::imread(imagePath);
	if (!img.empty())
	{
		//Make the image square by cutting away the sides
		int x = img.size().width < img.size().height ? 0 : (img.size().width - img.size().height) / 2;
		int y = img.size().width > img.size().height ? 0 : (img.size().height - img.size().width) / 2;
		int dim = img.size().width < img.size().height ? img.size().width : img.size().height;
		Rect myROI(x, y, dim, dim);
		Mat croppedImage = img(myROI);

		// Scale the image
		for (int scaleIndex = 0; scaleIndex < (sizeof(scales)/sizeof(*scales)); scaleIndex++)
		{
			double scale = scales[scaleIndex];
			int startX = ((1-scale)*croppedImage.size().width) / 2;
			int startY = ((1-scale)*croppedImage.size().height) / 2;
			Rect myROI2(startX, startY, scale*croppedImage.size().width, scale * croppedImage.size().height);
			Mat scaledImg = croppedImage(myROI2);
				
			//Rescale to 64x64
			resize(scaledImg, scaledImg, outputSize);

			// Rotate the image
			for (int rotIndex = 0; rotIndex < (sizeof(rotDegrees)/sizeof(*rotDegrees)); rotIndex++)
			{
				int angle = rotDegrees[rotIndex];
				Point2f srcCenter(scaledImg.cols/2.0F, scaledImg.rows/2.0F);
				Mat rotMat = getRotationMatrix2D(srcCenter, angle, 1.0);
				Mat rotScaledImg;
				warpAffine(scaledImg, rotScaledImg, rotMat, scaledImg.size());

				// Blurr the image with Gaussian
				for (int gaussianIndex = 0; gaussianIndex < (sizeof(guasianFilterSizes)/sizeof(*guasianFilterSizes)); gaussianIndex++)
				{
					//Multiple gradients of blurring
					Mat scaledRotBlurredImg;
					GaussianBlur( rotScaledImg, scaledRotBlurredImg, Size(guasianFilterSizes[gaussianIndex], guasianFilterSizes[gaussianIndex]), 0, 0 );

					//Store image
					//String outputFolder = robotImageInputCount%4 != 0 ? "Training/" : "Validation/";
					String outputFolder = "Training/";
					ostringstream outputName;
    				outputName << outputDatabasePath << outputFolder << "robot/" << robotImageNr++ << ".jpg";
					imwrite(outputName.str(), scaledRotBlurredImg);
				}
			}
		}
		robotImageInputCount++;
	}
}