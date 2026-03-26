// Name: Fabian Gonzalez
// GPU random walk. 
// nvcc P_GPURandomWalk.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 2,000 random walks of 10,000 steps simultaneously on the GPU, each with a different seed.
    Print the final positions of random walks 5, 100, 789, and 1622 as a spot check to get a warm 
    fuzzy feeling that your code is producing different random walks for each thread.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h>

// Defines
#define WALKS 2000

// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;
int *FinalPositionsX, *FinalPositionsY;
curandState *d_states;
dim3 GridSize;
dim3 BlockSize;

// Function prototypes
int getRandomDirection();
int main(int, char**);
void allocateMemory(); //Thius function will create the unified memory and the memory of two arrays I will use.
void setUpCudaDevices();
__device__ int getRandomDirectionCUDA();
__global__ void setupRNG(curandState* , unsigned long);
__global__ void randomWalk(curandState*, int*, int* , int); //Random walk simulation function

int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

//Memory Allocation
void allocateMemory()
{
    //CUDA Memory Allocation
    cudaMallocManaged(&FinalPositionsX, WALKS*sizeof(int));
    cudaMallocManaged(&FinalPositionsY, WALKS*sizeof(int));

    //Now, allocate the memory for all the different states generated
    cudaMalloc(&d_states, WALKS*sizeof(curandState));
}

//CUDA Device Set Ups
void setUpCudaDevices()
{
    BlockSize.x = 1024;
    BlockSize.y = 1;
    BlockSize.z = 1;

    GridSize.x = ((WALKS - 1)/(BlockSize.x)) + 1;
    GridSize.y = 1;
    GridSize.z = 1;
}

//CUDA Random Seed Generator
__global__ void setupRNG(curandState *state, unsigned long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < WALKS)
    {
        //To explain, here, every thread gets its own seed which they will get in the "state" array
        curand_init(seed, id, 0, &state[id]);
    }
}

//CUDA Step function
__device__ int getRandomDirectionCUDA(curandState *localState)
{
   	float randomNumber = curand_uniform(localState);
	
	if(randomNumber < 0.5f) return(-1);
	else return(1); 
}

//CUDA Random Walk Function
__global__ void randomWalk(curandState *state, int* FinalPositionsX, int *FinalPositionsY, int steps)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;

    //Only work if you are not bigger than number of walks.
    if(id < WALKS)
    {   
        //Load State Into Register
        curandState localState = state[id];

        //Statr of Simulation
        //Get variable positions
        int posX = 0;
        int posY = 0;

        //Do the random Walk
        for (int i = 0; i < steps; i++)
            {
            posX += getRandomDirectionCUDA(&localState);
            posY += getRandomDirectionCUDA(&localState);
            }

        //Store your final position
        FinalPositionsX[id] = posX;
        FinalPositionsY[id] = posY;

        //Return back state to global memory
        state[id] = localState;
    }
}

int main(int argc, char** argv)
{
	srand(time(NULL));

    //Initialize CUDA
    allocateMemory();
    setUpCudaDevices();
	
	printf("RAND_MAX for this implementation is = %d \n", RAND_MAX);
	
	int positionX = 0;
	int positionY = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection();
		positionY += getRandomDirection();
	}

	printf("\nFinal position = (%d,%d) \n", positionX, positionY);

    //CUDA
    //Initialize RNG
    setupRNG<<<GridSize, BlockSize>>>(d_states, time(NULL));

    //Run Simulation
    randomWalk<<<GridSize, BlockSize>>>(d_states, FinalPositionsX, FinalPositionsY, NumberOfRandomSteps);

    //Synchronize Everything
    cudaDeviceSynchronize();

    //Check Spots (from array)
    printf("\nGPU Final Location Walk 5: (%d, %d)", FinalPositionsX[4], FinalPositionsY[4]);
    printf("\nGPU Final Location Walk 100: (%d, %d)", FinalPositionsX[99], FinalPositionsY[99]);
    printf("\nGPU Final Location Walk 789: (%d, %d)", FinalPositionsX[788], FinalPositionsY[788]);
    printf("\nGPU Final Location Walk 1622: (%d, %d)\n", FinalPositionsX[1621], FinalPositionsY[1621]);

    //Clean Up Shop
    cudaFree(FinalPositionsX);
    cudaFree(FinalPositionsY);
    cudaFree(d_states);

	return 0;
}


