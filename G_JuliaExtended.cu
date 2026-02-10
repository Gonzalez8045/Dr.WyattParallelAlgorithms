// Name:
// Not simple Julia Set on the GPU
// nvcc G_JuliaExtended.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 However, it currently only runs on a 1024x1024 window.

 Your tasks:
 - Modify the code so it works on any given window size.
 - Add color to the fractal — be creative! You will be judged on your artistic flair.
   (Don't cut off your ear or anything, but try to make Vincent wish he'd had a GPU.)
*/

/*
 Purpose:
 To have some fun with your new GPU skills!
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <math.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1920;
unsigned int WindowHeight = 1024;

dim3 blockSize;
dim3 gridSize;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void setUpDevices();
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float *, float, float, float, float, float, float, int);

//Set up the blocks that will be used, as well as the grid size:
void setUpDevices()
{
	//Block size will remain constant in both x and y, since making them determine the window size will crash the program
	blockSize.x = 32;
	blockSize.y = 32;
	blockSize.z = 1;
	
	//Blocks in a grid
	gridSize.x = (WindowWidth + 31)/32;
	gridSize.y = (WindowHeight + 31)/32;
	gridSize.z = 1;
}

//Cuda error check
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

__global__ void colorPixels(float *pixels, float xMin, float xMax,float yMin, float yMax, float dx, float dy, int WW, int WH) 
{
	float x,y,XminGPU,YminGPU,mag,tempX;
	int count, idx, idy, pixelId, id;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	
	//Getting the offset into the pixel buffer. 
	//We need two indexes, x and y, in order to identify where in the universe we are
	idx = (blockIdx.x*blockDim.x + threadIdx.x);
	idy = (blockIdx.y*blockDim.y + threadIdx.y);

	//This gives me the pixel index
	pixelId = (idy*WW) + idx;

	//which, in turn, helps us find where in the array of pixels we are
	//We multiply by 3 since we need RGB for the gl function
	id = 3*pixelId;

	/*Each block will have its unique Xmin and Ymin
	The reason we need to multiply by 32 is to not double count the values from the prior block
	*/
	XminGPU = xMin + 32*dx*blockIdx.x;
	YminGPU = yMin + 32*dy*blockIdx.y;

	//Asigning each thread its x and y value of its pixel.
	//The change in X and Y remains unchanged from the thread. 
	x = XminGPU + dx*threadIdx.x;
	y = YminGPU + dy*threadIdx.y;

	//We need the "Don't give me stupid data" command
	//This is determined if the value of x is larger than what the maximum X is equal to
	if(idx >= WW || idy >= WH)
	{
		return; //do nothing if you break my dang bounds
	}
	
	//The count and magnitude equations don't change at all, yay!
	count = 0;
	mag = sqrt(x*x + y*y);;

	//Now do Julia Set Shenaningans
	while (mag < maxMag && count < maxCount) 
	{
		//We will be changing the x but we need its old value to find y.	
		tempX = x; 
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	
	// Smooth escape value normalized to [0,1]
    float t;
    if(count < MAXITERATIONS)
        t = count + 1 - log2f(log2f(mag)) / log2f(2.0f);
    else
        t = MAXITERATIONS;
    t /= MAXITERATIONS;

    // Rainbow coloring for pixels that escaped
	if(count < MAXITERATIONS) {
    	pixels[id]     = 0.5f + 0.5f * cosf(6.28318f * t + 0.0f);      // Red
    	pixels[id + 1] = 0.5f + 0.5f * cosf(6.28318f * t + 2.0944f);   // Green
    	pixels[id + 2] = 0.5f + 0.5f * cosf(6.28318f * t + 4.18879f);  // Blue
	} else {
    	// Inside Julia set → black
    	pixels[id]     = 0.0f;
    	pixels[id + 1] = 0.0f;
    	pixels[id + 2] = 0.0f;
}
}

void display(void) 
{ 
	//pixel arrays
	float *pixelsCPU, *pixelsGPU;

	//step sizes
	float stepSizeX, stepSizeY;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixelsCPU = (float *)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&pixelsGPU,WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);

	//initialize cuda here
	setUpDevices();
	cudaErrorCheck(__FILE__, __LINE__);	

	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, XMax, YMin, YMax, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(pixelsCPU, pixelsGPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU); 
	glFlush(); 
}

int main(int argc, char** argv)
{ 
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}