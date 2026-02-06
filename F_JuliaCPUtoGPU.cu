// Name: Fabian Gonzalez
// Simple Julia CPU.
// nvcc F_JuliaCPUtoGPU.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

float *Pixels, *PixelsGPU;

//definitions that will be used for GPU
#define N (WindowWidth * WindowHeight) //The amount of Pixels that will be used for the picture
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char*, int);
void allocateMemory(void);
void setUpDevices(void);
void cleanUp(void);
__global__ void cudaEscapeOrNotColor(float*, float, float, float, float, int, int, float);

//Check if Cuda did not destroy itself
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

//Allocate memory for both CPU and GPU

void allocateMemory(void)
{
    //Memory for CPU pixels
    Pixels = (float*)malloc(3*N*sizeof(float));
    
    //Memory for GPU pixels
    cudaMalloc(&PixelsGPU, 3*N*sizeof(float));
}

void setUpDevices(void)
{
    //BlockSize, will be used for X location of pixels
    BlockSize.x = WindowWidth;
    BlockSize.y = 1;
    BlockSize.z = 1;

    //GridSize, will be used for Y location of pixels
    GridSize.x = WindowHeight;
    GridSize.y = 1;
    GridSize.z = 1;
}

void cleanUp(void)
{
    free(Pixels);
    cudaFree(PixelsGPU);
}

__global__ void cudaEscapeOrNotColor(float* PixelsGPU, float XMin, float XMax, float YMin, float YMax, int n, int MaxCounts, float MaxMagnitude)
{
    //What is the data in the pixel that you will work on
    int px = threadIdx.x;
    int py = blockIdx.x;

    //tell me the id of the array that you will be working on;
    int id = 3*(py*blockDim.x + px);

    //Check that you are not working on a pixel we do now own;
    if((px >= (n/gridDim.x) || (py >= (n/blockDim.x))))
    {
        return;
    }

    //Define the step that you are (aka, the value you will be operating on)
    float stepSizeX = (XMax - XMin)/(float(n/gridDim.x));
    float stepSizeY = (YMax - YMin)/(float(n/blockDim.x));

    //Now, what x and y are you?
    float x = XMin + px*(stepSizeX);
    float y = YMin + py*(stepSizeY);

    //This is what you will do with that knowledge
    //First, you will define a value for the steps you will take
    int counts;

    //Then, you will keep in mind the value of x you started with, you will need it, as well as the values of the x and y squared
    float tempX;
    float xx, yy;

    //after, you will do the operations of our julia set
    for(counts = 0; counts < MaxCounts; counts++)
    {
        xx = x*x;
        yy = y*y;

        if(xx + yy > MaxMagnitude*MaxMagnitude)
        {
            break;
        }

        tempX = x;
        x = xx - yy + A;
        y = (2.0 * tempX * y) + B;
    }

    if(counts >= MaxCounts)
    {
        PixelsGPU[id] = 1.0f;
        PixelsGPU[id+1] = 0.0f;
        PixelsGPU[id+2] = 0.0f;
    }
    else
    {
        PixelsGPU[id] = 0.0f;
        PixelsGPU[id+1] = 0.0f;
        PixelsGPU[id+2] = 0.0f;
    }
    
}
/*
float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}
*/

void display(void) 
{ 
/*
	float *pixels; 
	float x, y, stepSizeX, stepSizeY;
	int k;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixels = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	k=0;
	y = YMin;
	while(y < YMax) 
	{
		x = XMin;
		while(x < XMax) 
		{
			pixels[k] = escapeOrNotColor(x,y);	//Red on or off returned from color
			pixels[k+1] = 0.0; 	//Green off
			pixels[k+2] = 0.0;	//Blue off
			k=k+3;			//Skip to next pixel (3 float jump)
			x += stepSizeX;
		}
		y += stepSizeY;
	}
*/

    //Get this man some devices
    setUpDevices();
    cudaErrorCheck(__FILE__, __LINE__);

    //Get this man some memory
    allocateMemory();
    cudaErrorCheck(__FILE__, __LINE__);

    //please cuda I need this, my CPU is kinda homeless
    cudaEscapeOrNotColor<<<GridSize,BlockSize>>>(PixelsGPU, XMin, XMax, YMin, YMax, N, MAXITERATIONS, MAXMAG);
    cudaErrorCheck(__FILE__, __LINE__);

    //Wait for the GPU, CPU
    cudaDeviceSynchronize();
    cudaErrorCheck(__FILE__, __LINE__);

    //Give what you got to the CPU, GPU
    cudaMemcpy(Pixels, PixelsGPU, 3*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaErrorCheck(__FILE__, __LINE__);

	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, Pixels); 
	glFlush();

    //Clean up your mess
    cleanUp();
    cudaErrorCheck(__FILE__, __LINE__); 
}

int main(int argc, char** argv)
{ 

    //Paint the Picture
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();

}