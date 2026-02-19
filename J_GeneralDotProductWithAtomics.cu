// Name:
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

//Header Files
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

//Pre-compile variables
#define N 100000000          // Original vector length
#define BLOCK_SIZE 1024      // Threads per block
#define TOLERANCE 0.01       // Percent error tolerance

// CPU and GPU pointers
float *A_CPU, *B_CPU, *C_CPU;
float *A_GPU, *B_GPU, *C_GPU;
float DotCPU, DotGPU;

// Grid and block sizes
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char*, int);
unsigned int nextPowerOf2(unsigned int N);
void allocateMemory(int paddedN);
void initializeVectors(int N, int paddedN);
void dotProductCPU(float*, float*, int);
long elapsedTime(struct timeval, struct timeval);
bool check(float cpuVal, float gpuVal, float tolerance);
int chooseBestDevice(int paddedN);

// CUDA kernel
__global__ void dotProductGPU(float *a, float *b, float *c, int n) {
    __shared__ float c_sh[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    c_sh[tid] = (idx < n) ? (a[idx] * b[idx]) : 0.0f;
    __syncthreads();

    int fold = blockDim.x / 2;
    while (fold > 0) {
        if (tid < fold)
            c_sh[tid] += c_sh[tid + fold];
        __syncthreads();
        fold /= 2;
    }

    if (tid == 0)
        atomicAdd(c, c_sh[0]);
}

// Check CUDA errors
void cudaErrorCheck(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s | File: %s | Line: %d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}

// Next power of 2 function
unsigned int nextPowerOf2(unsigned int N) {
    if (N == 0) return 1;
    double exp = ceil(log2((double)N));
    return 1 << (unsigned int)exp;
}

// Allocate memory on CPU and GPU
void allocateMemory(int paddedN) {
    A_CPU = (float*)malloc(paddedN * sizeof(float));
    B_CPU = (float*)malloc(paddedN * sizeof(float));
    C_CPU = (float*)malloc(sizeof(float));

    cudaMalloc(&A_GPU, paddedN * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&B_GPU, paddedN * sizeof(float));
    cudaErrorCheck(__FILE__, __LINE__);
    cudaMalloc(&C_GPU, sizeof(float)); // only need one float
    cudaErrorCheck(__FILE__, __LINE__);
}

// Initialize vectors and zero-pad
void initializeVectors(int N, int paddedN) {
    for (int i = 0; i < N; i++) {
        A_CPU[i] = (float)i;
        B_CPU[i] = (float)(3 * i);
    }
    for (int i = N; i < paddedN; i++) {
        A_CPU[i] = 0.0f;
        B_CPU[i] = 0.0f;
    }
}

// CPU dot product
void dotProductCPU(float *a, float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    DotCPU = sum;
}

// Time difference in microseconds
long elapsedTime(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

// Check percent error
bool check(float cpuVal, float gpuVal, float tolerance) {
    double percentError = fabs((gpuVal - cpuVal) / cpuVal) * 100.0;
    printf("Percent error: %lf%%\n", percentError);
    return (percentError < tolerance);
}

// Choose the best device
int chooseBestDevice(int paddedN) {
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) return -1;

    int best = -1;
    int maxThreads = 0;

    for (int i = 0; i < count; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);

        int totalThreads = p.maxThreadsPerMultiProcessor * p.multiProcessorCount;
        if (totalThreads < paddedN) continue;
        if (p.major < 3) continue; // atomicAdd on float requires compute capability 3.0+

        if (totalThreads > maxThreads) {
            maxThreads = totalThreads;
            best = i;
        }
    }
    return best;
}

// Cleanup
void cleanUp(int paddedN) {
    free(A_CPU); free(B_CPU); free(C_CPU);
    cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
}

int main() {
    struct timeval start, end;
    long timeCPU, timeGPU;

    unsigned int vectorSize = N;
    unsigned int paddedSize = nextPowerOf2(vectorSize);

    int bestDevice = chooseBestDevice(paddedSize);
    if(bestDevice == -1) {
        printf("No suitable GPU found.\n");
        return 1;
    }

    cudaSetDevice(bestDevice);
    int currentDevice;
    cudaGetDevice(&currentDevice);
    printf("Using GPU device %d\n", currentDevice);

    // Grid/block
    BlockSize.x = BLOCK_SIZE; BlockSize.y = 1; BlockSize.z = 1;
    GridSize.x = (paddedSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GridSize.y = 1; GridSize.z = 1;

    allocateMemory(paddedSize);
    initializeVectors(vectorSize, paddedSize);

    // CPU dot product
    gettimeofday(&start, NULL);
    DotCPU = 0.0f;
    for(int i=0;i<paddedSize;i++) DotCPU += A_CPU[i]*B_CPU[i];
    gettimeofday(&end, NULL);
    timeCPU = elapsedTime(start, end);

    // GPU dot product
    cudaMemset(C_GPU, 0, sizeof(float));
    cudaMemcpy(A_GPU, A_CPU, paddedSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_GPU, B_CPU, paddedSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    gettimeofday(&start, NULL);
    dotProductGPU<<<GridSize, BlockSize>>>(A_GPU, B_GPU, C_GPU, paddedSize);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(&DotGPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost);

    timeGPU = elapsedTime(start, end);

    printf("CPU: %f | GPU: %f\n", DotCPU, DotGPU);
    if(check(DotCPU, DotGPU, TOLERANCE)) printf("GPU result is correct!\n");
    else printf("GPU result is incorrect!\n");

    printf("CPU time: %ld us | GPU time: %ld us\n", timeCPU, timeGPU);
    cleanUp();
    return 0;
}
