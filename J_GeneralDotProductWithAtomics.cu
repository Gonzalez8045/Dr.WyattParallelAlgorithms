// Name: Fabian GonzalezS
// Robust Vector Dot product 
// nvcc J_GeneralDotProductWithAtomics.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors of any length using shared memory to
 reduce the number of global memory accesses. However, since blocks can’t synchronize
 with each other, the final reduction must be handled on the CPU.

 To simplify the GPU-side logic, we’ll add some “pregame” setup and use atomic adds.

 1. Make sure the number of threads per block is a power of 2. This avoids messy edge
    cases during the reduction step. If it’s not a power of 2, print an error message
    and exit. (Without this, you'd have to check if the reduction is even or not,
    add the last element to the first, adjust the loop, etc.)

 2. Calculate the correct number of blocks needed to process the entire vector.
    Then check device properties to ensure the grid and block sizes are within hardware limits.
    Just because it works on your fancy GPU doesn’t mean it will work on your client’s older one.
    If the block or grid size exceeds the device’s capabilities, report the issue and exit gracefully.

 3. It’s inefficient to check inside your kernel if a thread is working past the end of the vector
    on every iteration. Instead, figure out how many extra elements are needed to fill out the grid,
    and pad the vector with zeros. Zero-padding doesn’t affect the dot product (0 * anything = 0).
    Use `cudaMemset` to explicitly zero out your device memory — don’t rely on "getting lucky"
    like you might have in previous assignments.

 4. In previous assignments, we had to do the final reduction on the CPU because we couldn't sync blocks.
    Now, use **atomic adds** to sum partial results directly on the GPU and avoid CPU post-processing.
    Then, copy the final result back to the CPU using `cudaMemcpy`.

    Note: Atomic operations on floats are only supported on GPUs with compute capability 3.0 or higher.
    Use device properties to check this before running the kernel.
    While you’re at it, if multiple GPUs are available, select the best one based on compute capability.

 5. Add any additional bells and whistles to make your code more robust and user-proof.
    Think of edge cases or bad input your client might provide and handle it cleanly.
*/

/*
 Purpose:
 To learn how to use atomic adds to avoid jumping out of the kernel for block synchronization.
 This is also your opportunity to make the code "foolproof" — handling edge cases gracefully.

 At this point, you should understand all the CUDA basics.
 From now on, we’ll focus on refining that knowledge and adding advanced features.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Defines
#define N 100000 // Length of the vector
#define BLOCK_SIZE 1024 // Threads in a block

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *resultGPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices(int N);
void allocateMemory(int, int);
void innitialize();
void dotProductCPU(float*, float*, float*, int);
int nextPowerOfTwo(int N);
void padding(int, int);
__global__ void dotProductGPU(float*, float*, float*, int);
int bestGPU(int N);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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


//This function will check the next power of two.
int nextPowerOfTwo(int N)
{
	//Power and Result for equation
    int power = 1;
	int result = 1;

    //Now, we do a while loop to check for the next value, which will compare the power of two next to the vector number N
    while(result < N)
    {
        //Our result will be 1 at the beggining (2^0 = 1)
        

        //We set up a for loop using our power, and will increment it after each iteration
        for(int i = 0; i < power; i++)
        {
            result *= 2; //This computes the power
        }

        power++; //If the prior power didn't work, incremnet it and check again against N, else, do all the prior operation
    }

    return result; //This will gives us the next power of 2.
}

//This function will padd our GPU vectors;
void padding(int N, int pN)
{
        cudaMemset(A_GPU + N, 0, (pN - N)*sizeof(float));
        cudaMemset(B_GPU + N, 0, (pN - N)*sizeof(float));
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	int threadIndex = threadIdx.x;
	int vectorIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ float c_sh[BLOCK_SIZE];
	
	    c_sh[threadIndex] = (a[vectorIndex] * b[vectorIndex]);
	__syncthreads();
	
	int fold = blockDim.x;
	while(1 < fold)
	{
		if(fold%2 != 0)
		{
			if(threadIndex == 0 && (vectorIndex + fold - 1) < n)
			{
				c_sh[0] = c_sh[0] + c_sh[0 + fold - 1];
			}
			fold = fold - 1;
		}
		fold = fold/2;
		if(threadIndex < fold && (vectorIndex + fold) < n)
		{
			c_sh[threadIndex] = c_sh[threadIndex] + c_sh[threadIndex + fold];
			
		}
		__syncthreads();
	}
	
    if(threadIndex == 0)
        atomicAdd(c, c_sh[threadIndex]);
}

//This function will decide the best GPU to use, comparing major and minors
int bestGPU(int N)
{
    //Check how many devices you have
    int deviceCount, bestGPU, bestMajor, bestMinor;
    cudaGetDeviceCount(&deviceCount);

    //Assume none are good at first
    bestGPU = -1;
    bestMajor = 0;
    bestMinor = 0;

    for(int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        //Get your major and minor from properties
        int propMajor, propMinor;
        propMajor = prop.major;
        propMinor = prop.minor;

        //First, check that the major is greater than 3 (so it can do floats)
        if(propMajor < 3)
            continue; //This just says "if you can't do floats, shut up"

        //Now, if they pass that check, see if they have enough threads to do the operation
        int threadCount = prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount;

        if(threadCount < N)
            continue;
        
        //After this check if you have a better major and minor:
        if(propMajor > bestMajor)
        {
            if(propMinor > bestMinor)
            {
                bestGPU = i; //If you do, you are prolly the better GPU.
            }
        }
    }

    return bestGPU; //return whatever value you get.
}

// This will be the layout of the parallel space we will be using.
void setUpDevices(int N)
{
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory(int N, int pN)
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,pN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,pN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&resultGPU, sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will doting.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = fabs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(resultGPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
    //We need to set up the values for N and the padded N, which will be integer values
    int vectorNum = N;
    int paddedVec = nextPowerOfTwo(N);

	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	

    //Now, check which GPU is the best, before setting it:
    int usedGPU = bestGPU(paddedVec);
    
    //If you don't have a GPU that can do this, this program can't be run.
    if(usedGPU == -1)
    {
        printf("This operation cannot be performed\n");
        printf("It might be time to upgrade\n");
        printf("Goodbye");
        exit(0);
    }

	// Setting up the GPU
	setUpDevices(paddedVec);
	
	// Allocating the memory you will need.
	allocateMemory(vectorNum, paddedVec);
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, resultGPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(&DotGPU, resultGPU, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}