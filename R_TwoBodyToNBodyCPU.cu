// Name:
// Two body problem
// nvcc R_TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0
#define MASS 1.0
#define N 20

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
float *positionX, *positionY, *positionZ, *masses; //Position Vecotor
float *velocityX, *velocityY, *velocityZ; //Velocity Vector 
float *forceX, *forceY, *forceZ; 
float *colorR, *colorG, *colorB;

// Function prototypes
void allocateMemory();
void set_initial_conditions();
void Drawwirebox();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

void allocateMemory()
{
    //Position Memory Allocation
    positionX = (float*)malloc(N*sizeof(float)); 
    positionY = (float*)malloc(N*sizeof(float)); 
    positionZ = (float*)malloc(N*sizeof(float));

    //Velocity Memory Allocation
    velocityX = (float*)malloc(N*sizeof(float));
    velocityY = (float*)malloc(N*sizeof(float));
    velocityZ = (float*)malloc(N*sizeof(float));

    //Force Memory Allocation
    forceX = (float*)malloc(N*sizeof(float));
    forceY = (float*)malloc(N*sizeof(float));
    forceZ = (float*)malloc(N*sizeof(float));

    //Mass Memory Allocation
    masses = (float*)malloc(N*sizeof(float));

    //Color Matrix Memory Allocaton
    colorR = (float*)malloc(N*sizeof(float));
    colorG = (float*)malloc(N*sizeof(float));
    colorB = (float*)malloc(N*sizeof(float));
}

void set_initial_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	float dx, dy, dz, separation;
	
    positionX[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
    positionY[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
    positionZ[0] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;

    for(int i = 1; i < N; i++)
    {
        //This will be here in case the position is invalid to start with
        retry:

        //Get Your Position
        positionX[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
        positionY[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
        positionZ[i] = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;

        for(int j = 0; j < i; j++)
        {
            //Check your differences with each existing guy
            dx = positionX[i] - positionX[j];
            dy = positionY[i] - positionY[j];
            dz = positionZ[i] - positionZ[j];

            separation = sqrt((dx*dx) + (dy*dy) + (dz*dz));

            //If you are inside any of the spheres, get another position
            if(separation < DIAMETER) goto retry;
        
        }
    }
	

    for(int k = 0; k < N; k++)
    {
        //Set the velocities
        velocityX[k] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
        velocityY[k] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
        velocityZ[k] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;

        masses[k] = MASS; //Get your mass

        colorR[k] = (float)rand() / RAND_MAX;
        colorG[k] = (float)rand() / RAND_MAX;
        colorB[k] = (float)rand() / RAND_MAX;
    }
}

//This function remains the same, yay!
void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
    for(int i = 0; i < N; i++)
    {
        if(positionX[i] > halfBoxLength)
	    {
		    positionX[i] = 2.0*halfBoxLength - positionX[i];
		    velocityX[i] = - velocityX[i];
	    }
	    else if(positionX[i] < -halfBoxLength)
	    {
		    positionX[i] = -2.0*halfBoxLength - positionX[i];
		    velocityX[i] = - velocityX[i];
	    }
	
	    if(positionY[i] > halfBoxLength)
	    {
		    positionY[i] = 2.0*halfBoxLength - positionY[i];
		    velocityY[i] = - velocityY[i];
	    }
	    else if(positionY[i] < -halfBoxLength)
	    {
		    positionY[i] = -2.0*halfBoxLength - positionY[i];
		    velocityY[i] = - velocityY[i];
	    }
			
	    if(positionZ[i] > halfBoxLength)
	    {
		    positionZ[i] = 2.0*halfBoxLength - positionZ[i];
		    velocityZ[i] = - velocityZ[i];
	    }
	    else if(positionZ[i] < -halfBoxLength)
	    {
		    positionZ[i] = -2.0*halfBoxLength - positionZ[i];
		    velocityZ[i] = - velocityZ[i];
	    }
    }
}

void get_forces()
{
    //Clean old forces for simulation
    for(int i = 0; i < N; i++)
    {
        forceX[i] = 0;
        forceY[i] = 0;
        forceZ[i] = 0;
    }
    
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;
	
    for(int i = 0; i < N; i++)
    {
        for(int j = i+1; j < N; j++)
        {
            //Calculates distance between the pooint
            dx = positionX[i] - positionX[j];
            dy = positionY[i] - positionY[j];
            dz = positionZ[i] - positionZ[j];

            //Get the changes in Velocity
            dvx = velocityX[i] - velocityX[j];
            dvy = velocityY[i] - velocityY[j];
            dvz = velocityZ[i] - velocityZ[j];
            inout = dx*dvx + dy*dvy + dz*dvz;

            //Calculate Distance
            r2 = dx*dx + dy*dy + dz*dz;
            r = sqrt(r2) + 0.000001; //The little factor is just to make sure we are not dividing by zero.

            //calculate the magnitude of the force
            forceMag = (masses[i] * masses[j] * GRAVITY) / r2;

            //Check "pushback" from other spheres
            if (r < DIAMETER)
            {
                // If moving toward each other (inout <= 0), push hard
                if(inout <= 0.0)
                {
                    forceMag += SPHERE_PUSH_BACK_STRENGTH * (r - DIAMETER);
                }
                // If already moving away, push less to prevent "sticking"
                else
                {
                    forceMag += PUSH_BACK_REDUCTION * SPHERE_PUSH_BACK_STRENGTH * (r - DIAMETER);
                }
            }
            // 3. Apply the forces (Final Settings)
            // Divide by r to "normalize" the direction
            forceX[i] -= forceMag * dx / r;
            forceY[i] -= forceMag * dy / r;
            forceZ[i] -= forceMag * dz / r;

            // Newton's 3rd Law: Equal and opposite reaction for sphere j
            forceX[j] += forceMag * dx / r;
            forceY[j] += forceMag * dy / r;
            forceZ[j] += forceMag * dz / r;
        }
    }
}

void move_bodies(float time)
{
	if(time == 0.0)
	{
        for(int i = 0; i < N; i++)
        {
		    velocityX[i] += 0.5*DT*(forceX[i] - DAMP*velocityX[i])/masses[i];
		    velocityY[i] += 0.5*DT*(forceY[i] - DAMP*velocityY[i])/masses[i];
		    velocityZ[i] += 0.5*DT*(forceZ[i] - DAMP*velocityZ[i])/masses[i];
        }
	}
	else
	{
        for(int i = 0; i < N; i++)
        {
		    velocityX[i] += DT*(forceX[i] - DAMP*velocityX[i])/masses[i];
		    velocityY[i] += DT*(forceY[i] - DAMP*velocityY[i])/masses[i];
		    velocityZ[i] += DT*(forceZ[i] - DAMP*velocityZ[i])/masses[i];
        }
	}

    for(int i = 0; i < N; i++)
    {
       positionX[i] += DT*velocityX[i];
       positionY[i] += DT*velocityY[i];
       positionZ[i] += DT*velocityZ[i];
    }
	
	keep_in_box();
}

void nbody()
{	
    static float time = 0.0;
	
	if(time < STOP_TIME)
	{
        get_forces();
        move_bodies(time);

        time += DT;

        glutPostRedisplay();
	}
    else
    {
        printf("\n DONE \n");
    }
}

void Display(void)
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	Drawwirebox();
	
    for(int i = 0; i < N; i++)
    {
        glPushMatrix();
            glColor3f(colorR[i], colorG[i], colorB[i]);
            glTranslatef(positionX[i], positionY[i], positionZ[i]);
            glutSolidSphere(radius, 20, 20);
        glPopMatrix();
    }
	
	glutSwapBuffers();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
    //Initialize variables:
    allocateMemory();
    set_initial_conditions();

    //Work those glutes
    glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutIdleFunc(nbody);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
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
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
