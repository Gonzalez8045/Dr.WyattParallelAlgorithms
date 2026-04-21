// Name:
// nBody code on multiple GPUs. 
// nvcc U_1GPU_To_2GPUs_NBody.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some robust N-body code that runs on the GPU with all the bells and whistles removed. 
 Modify it so it runs on two GPUs.
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define BLOCK_SIZE 128
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
#define RUN_TIME 10.0

// Globals
int N, deviceCount, trueCount;
float3 *P, *V, *F;
float *M; 
float3 **PGPU, **VGPU, **FGPU;
float **MGPU;
float GlobeRadius, Diameter, Radius;
float Damp;
dim3 BlockSize;
dim3 GridSize;

// Function prototypes
void cudaErrorCheck(const char *, int);
void drawPicture();
void setup();
__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int total_n, int local_n, int offset);
__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int local_n, int offset);
void nBody();

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

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
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

void setup()
{
	float randomAngle1, randomAngle2, randomRadius;
	float d, dx, dy, dz;
	int test;

	cudaGetDeviceCount(&deviceCount);
	
	N = 1000;
	
	Damp = 0.5;
	
	M = (float*)malloc(N*sizeof(float));
	P = (float3*)malloc(N*sizeof(float3));
	V = (float3*)malloc(N*sizeof(float3));
	F = (float3*)malloc(N*sizeof(float3));

	//Making pointer pointers for the array of values in the GPU
	PGPU = (float3**)malloc((deviceCount) * sizeof(float3*));
	VGPU = (float3**)malloc((deviceCount) * sizeof(float3*));
	FGPU = (float3**)malloc((deviceCount) * sizeof(float3*));
	MGPU = (float**)malloc((deviceCount) * sizeof(float*));

	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	

	int batchSize = N/deviceCount;
	int remainder = N % deviceCount;
	int offset = 0;

	for(int i = 0; i < deviceCount; i++)
	{
		// 1. Calculate how many particles THIS GPU is responsible for updating
        int gpu_count = batchSize + (i < remainder ? 1 : 0);

        cudaSetDevice(i);

        // 2. ALLOCATE THE MASTER READ-ONLY LIST (Full N)
        // Every GPU needs this to calculate forces against every other particle
        cudaMalloc(&PGPU[i], N * sizeof(float3)); 

        // 3. ALLOCATE THE WORKER SLICES (Only gpu_count)
        // These are the specific velocities/forces this GPU will update
        cudaMalloc(&VGPU[i], gpu_count * sizeof(float3));
        cudaMalloc(&FGPU[i], gpu_count * sizeof(float3));
        cudaMalloc(&MGPU[i], N * sizeof(float));

        // 4. INITIAL COPIES
        // Push the full global positions to this GPU's master list
        cudaMemcpy(PGPU[i], P, N * sizeof(float3), cudaMemcpyHostToDevice);
		cudaMemcpy(MGPU[i], M, N * sizeof(float), cudaMemcpyHostToDevice);
        
        // Push the specific slice of V, F, and M
        cudaMemcpy(VGPU[i], &V[offset], gpu_count * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(FGPU[i], &F[offset], gpu_count * sizeof(float3), cudaMemcpyHostToDevice);
 

        offset += gpu_count;
	}

	// Define the grid and block sizes
    BlockSize.x = BLOCK_SIZE;
    BlockSize.y = 1;
    BlockSize.z = 1;

    // Calculate grid size based on the largest possible batch
    int maxBatch = batchSize + (remainder > 0 ? 1 : 0);
    GridSize.x = (maxBatch + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GridSize.y = 1;
    GridSize.z = 1;
}

__global__ void getForces(float3 *p, float3 *v, float3 *f, float *m, float g, float h, int total_n, int local_n, int offset)
{
    float dx, dy, dz, d, d2;
    float force_mag;
    
    // i is the LOCAL index (0 to 499)
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    // 1. CHANGE: Check against local_n, not total_n
    if(i < local_n)
    {
        // 2. global_i tells us where THIS particle is in the big 1000-element PGPU array
        int global_i = i + offset;

        f[i].x = 0.0f;
        f[i].y = 0.0f;
        f[i].z = 0.0f;
        
        for(int j = 0; j < total_n; j++) // Loop through everyone
        {
            if(global_i != j)
            {
                // Use the global_i to get "my" position from the big list
                dx = p[j].x - p[global_i].x;
                dy = p[j].y - p[global_i].y;
                dz = p[j].z - p[global_i].z;
                
                d2 = dx*dx + dy*dy + dz*dz;
                d  = sqrt(d2);
                
                // 3. INDEXING CHECK: 
                // m[i] is safe (local mass). 
                // BUT: m[j] is risky! (Wait, does the GPU have everyone's mass?)
                
                // --- CRITICAL NOTE ---
                // If you want to use m[j], every GPU needs a FULL copy of the mass array M,
                // just like they have a full copy of positions P. 
                // Otherwise, GPU 0 can't "see" the mass of particles on GPU 1.
                
                force_mag  = (g*m[i]*m[j])/(d2) - (h*m[i]*m[j])/(d2*d2);
                
                f[i].x += force_mag*dx/d;
                f[i].y += force_mag*dy/d;
                f[i].z += force_mag*dz/d;
            }
        }
    }
}

__global__ void moveBodies(float3 *p, float3 *v, float3 *f, float *m, float damp, float dt, float t, int local_n, int offset)
{   
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Use local_n to stay within the bounds of your v, f, and m slices
    if(i < local_n)
    {
        // 1. Update Velocity (Uses local slices f, v, and m)
        float inv_m = 1.0f / m[i]; // Minor optimization: multiply by reciprocal
        float multiplier = (t == 0.0f) ? dt / 2.0f : dt;

        v[i].x += ((f[i].x - damp * v[i].x) * inv_m) * multiplier;
        v[i].y += ((f[i].y - damp * v[i].y) * inv_m) * multiplier;
        v[i].z += ((f[i].z - damp * v[i].z) * inv_m) * multiplier;

        // 2. Update Position
        // We need to write to the correct spot in the N-sized 'p' array
        int global_i = i + offset; 

        p[global_i].x += v[i].x * dt;
        p[global_i].y += v[i].y * dt;
        p[global_i].z += v[i].z * dt;
    }
}

void nBody()
{
    float t = 0.0;
    int drawCount = 0;
    int batchSize = N / deviceCount;
    int remainder = N % deviceCount;

    while(t < RUN_TIME)
    {
        int offset = 0;

        // --- PHASE 1: LAUNCH KERNELS ON ALL GPUs ---
        for(int i = 0; i < deviceCount; i++)
        {
            cudaSetDevice(i);
            int gpu_count = batchSize + (i < remainder ? 1 : 0);

            // getForces calculates the N^2 interactions
            getForces<<<GridSize, BlockSize>>>(PGPU[i], VGPU[i], FGPU[i], MGPU[i], G, H, N, gpu_count, offset);
            
            // moveBodies updates positions and velocities locally
            moveBodies<<<GridSize, BlockSize>>>(PGPU[i], VGPU[i], FGPU[i], MGPU[i], Damp, DT, t, gpu_count, offset);
            
            offset += gpu_count;
        }

        // --- PHASE 2: SYNC (The "Gather" Step) ---
        // We must pull the NEW positions back to the CPU to create a complete 'P' array
        offset = 0;
        for(int i = 0; i < deviceCount; i++)
        {
            cudaSetDevice(i);
            int gpu_count = batchSize + (i < remainder ? 1 : 0);

            // Important: We only copy the specific SLICE that this GPU just updated
            cudaMemcpy(&P[offset], &PGPU[i][offset], gpu_count * sizeof(float3), cudaMemcpyDeviceToHost);
            
            offset += gpu_count;
        }

        // --- PHASE 3: BROADCAST (The "Update" Step) ---
        // Now that the CPU's 'P' is full and updated, we send it back to EVERY GPU's master list
        for(int i = 0; i < deviceCount; i++)
        {
            cudaSetDevice(i);
            cudaMemcpy(PGPU[i], P, N * sizeof(float3), cudaMemcpyHostToDevice);
        }

        // --- PHASE 4: DRAWING ---
        if(drawCount == DRAW_RATE) 
        {   
            // P is now fully updated on the Host, so drawPicture works perfectly
            drawPicture(); 
            drawCount = 0;
        }
        
        t += DT;
        drawCount++;
    }
}

int main(int argc, char** argv)
{
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Nbody Two GPUs");
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
	glutDisplayFunc(drawPicture);
	glutIdleFunc(nBody);
	
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
