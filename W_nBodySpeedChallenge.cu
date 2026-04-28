// Name: Fabian Gonzalez
// Optimizing nBody GPU code. 
// nvcc -O3 --use_fast_math W_nBodySpeedChallenge.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean n-body code that runs on the GPU for any number of bodies (within reason). Take this code and make it 
 run as fast as possible using any tricks you know or can find (Like using NVIDIA Nsight Systems). Keep the same general 
 format so we can time it and compare it with others' code. This will be a competition.
 
 First place: 20 extra points on this HW
 
 To focus more on new ideas rather than just using a bunch of if statements to avoid going out of bounds, N will be a power 
 of 2 and 256 < N < 262,144. Put a check in your code to make sure this is true. The code most run on any power of 2 bodies
 also the final picture most look close to the same as it did before the speedup or something went wrong in the code.

 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate.
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
 
 Use this code (before your changes) as the baseline code to check your nbody speedup.
*/

/*
 Purpose:
 To use what you have learned in this course to optimize code with the add of NVIDIA Nsight Systems.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand_kernel.h>

// Defines
#define BLOCK_SIZE 256
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0f
#define H 10.0f
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float4 *P, *V, *F; 
float4 *PGPU, *VGPU, *FGPU;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
__global__ void getForcesTiled(float4 *p, float4 *f, float g, float h, int n);
__global__ void moveBodies(float4 *p, float4 *v, float4 *f, float damp, float dt, float t, int n);
void nBody();
int main(int, char**);
__global__ void setupPositions(float4 *p, unsigned long seed, int n, float globeRadius);

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

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		printf("\n The simulation is running.\n");
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	cudaMemcpyAsync(P, PGPU, N * sizeof(float4), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    		nBody();
    		cudaDeviceSynchronize();
		cudaErrorCheck(__FILE__, __LINE__);
    	gettimeofday(&end, NULL);
    	drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    
    BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	GridSize.y = 1;
	GridSize.z = 1;
	
    Damp = 0.5;
    
    P = (float4*)malloc(N*sizeof(float4));
    V = (float4*)malloc(N*sizeof(float4));
    F = (float4*)malloc(N*sizeof(float4));
    
	cudaMalloc(&PGPU,N*sizeof(float4));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU,N*sizeof(float4));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU,N*sizeof(float4));
	cudaErrorCheck(__FILE__, __LINE__);
    	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
    setupPositions<<<blocksPerGrid, threadsPerBlock>>>(PGPU, time(NULL), N, GlobeRadius);

    cudaMemset(VGPU, 0, N * sizeof(float4));
    cudaMemset(FGPU, 0, N * sizeof(float4));    
	
	printf("\n To start timing go to the nBody window and type s.\n");
	printf("\n To quit type q in the nBody window.\n");
}

__global__ void getForcesTiled(float4 *p, float4 *f, float g, float h, int n)
{
    __shared__ float4 shPos[BLOCK_SIZE];
    
    int i = threadIdx.x + (blockIdx.x << 8); 
    float4 myPos = p[i];
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < gridDim.x; tile++)
    {
        // Load tile into shared memory
        shPos[threadIdx.x] = p[(tile << 8) + threadIdx.x];
        __syncthreads();

        // TRICK: Prefetch the first position into a register
        float4 otherPos = shPos[0];

        #pragma unroll 16
        for (int j = 0; j < BLOCK_SIZE; j++)
        {
            float dx = otherPos.x - myPos.x;
            float dy = otherPos.y - myPos.y;
            float dz = otherPos.z - myPos.z;
            
            float d2 = dx*dx + dy*dy + dz*dz + 1e-9f; 
            float invDist = rsqrtf(d2);
            float invDist2 = invDist * invDist;
            
            float s = (myPos.w * otherPos.w) * invDist2 * (g - h * invDist2) * invDist;
            
            acc.x += s * dx;
            acc.y += s * dy;
            acc.z += s * dz;

            // REGISTER PREFETCH: Load the NEXT position while the math for 's' is processing
            if (j < 255) otherPos = shPos[j + 1];
        }
        __syncthreads();
    }

    f[i] = make_float4(acc.x, acc.y, acc.z, 0.0f);
}

__global__ void moveBodies(float4 *p, float4 *v, float4 *f, float damp, float dt, float t, int n)
{   
    // No if(i < n) required!
    int i = threadIdx.x + (blockIdx.x << 8); 
    
    float4 pos = p[i];
    float4 vel = v[i];
    float4 force = f[i];
    
    // Mass is stored in the .w component
    float invMass = 1.0f / pos.w;

    // Use a multiplier to handle the t=0 (initial half-step) case without extra branching
    float timeMult = (t == 0.0f) ? 0.5f : 1.0f;

    // v = v + a * dt (incorporating damping)
    vel.x += ((force.x - damp * vel.x) * invMass) * dt * timeMult;
    vel.y += ((force.y - damp * vel.y) * invMass) * dt * timeMult;
    vel.z += ((force.z - damp * vel.z) * invMass) * dt * timeMult;

    // p = p + v * dt
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;

    // Write back updated values
    v[i] = vel;
    p[i] = pos;
}

__global__ void setupPositions(float4 *p, unsigned long seed, int n, float globeRadius)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
        curandState state;
        curand_init(seed, i, 0, &state);

        // Standard random spherical distribution
        float u = curand_uniform(&state); // for radius distribution
        float v = curand_uniform(&state); // for azimuthal angle
        float w = curand_uniform(&state); // for polar angle

        float phi = v * 2.0f * PI;
        float cosTheta = 2.0f * w - 1.0f;
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
        float r = globeRadius * powf(u, 1.0f/3.0f); // Uniform volume distribution

        p[i].x = r * sinTheta * cosf(phi);
        p[i].y = r * sinTheta * sinf(phi);
        p[i].z = r * cosTheta;
        p[i].w = 1.0f; // Mass stored in W
}

void nBody()
{
	int    drawCount = 0; 
	float  t = 0.0;
	float dt = 0.0001;

	while(t < RUN_TIME)
	{
		getForcesTiled<<<GridSize,BlockSize>>>(PGPU, FGPU, G, H, N);
		cudaErrorCheck(__FILE__, __LINE__);
		moveBodies<<<GridSize,BlockSize>>>(PGPU, VGPU, FGPU, Damp, dt, t, N);
		cudaErrorCheck(__FILE__, __LINE__);

		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag) 
			{	
				drawPicture();
			}
			drawCount = 0;
		}
		
		t += dt;
		drawCount++;
	}
}

int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
        if ((N & (N - 1)) != 0 || N < 256 || N > 262144) 
        {
            printf("Error: N must be a power of 2 between 256 and 262,144\n");
            exit(0);
        }
	}
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Challenge");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}





