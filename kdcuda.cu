#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<curand.h>
#include<curand_kernel.h>

#define TPB 1024 // Threads per block

// constant memory
__constant__ double ctes_d[28];
/*
ctes[0]  PI
ctes[1]  q
ctes[2]  m
ctes[3]  hbar
ctes[4]  c
ctes[5]  v0
ctes[6]  beta
ctes[7]  gamma
ctes[8]  omega_Compton
ctes[9]  omega_A
ctes[10] omega_B
ctes[11] k_Compton
ctes[12] k_A
ctes[13] k_B
ctes[14] lambda_Compton
ctes[15] omega_Laser
ctes[16] omega_A Laser
ctes[17] omega_B Laser
ctes[18] k_Laser
ctes[19] k_A Laser
ctes[20] k_B Laser
ctes[21] lambda_Laser
ctes[22] E0 Laser
ctes[23] E0 ZPF
ctes[24] dt
ctes[25] D
ctes[26] zimp
ctes[27] Delta
*/
double ctes[27];

__device__ double f(double eta1, double eta2, double t, double y, double z , double vz)
{
	double phi1zpf=ctes_d[9]*t-ctes_d[13]*z+eta1+ctes_d[10]*t-ctes_d[12]*z;
	double phi2zpf=ctes_d[9]*t-ctes_d[13]*z+eta2-ctes_d[10]*t+ctes_d[12]*z;
	double result=0.0;
	if (z<=ctes_d[25]){
		//double phi1L=ctes_d[16]*t-ctes_d[20]*z-ctes_d[18]*y;
		//double phi2L=ctes_d[16]*t-ctes_d[20]*z+ctes_d[18]*y;
		double phi1L=ctes_d[15]*t-ctes_d[18]*y;
		double phi2L=ctes_d[15]*t+ctes_d[18]*y;
		double EL=ctes_d[22]*exp(-(pow(z-ctes_d[25]/2.0,2.0))/(2.0*pow(ctes_d[25]/5.0,2.0)));
		result = ctes_d[1]*sqrtf(1-pow(ctes_d[6],2.0))*ctes_d[23]*(cos(phi1zpf)+cos(phi2zpf))/ctes_d[2]+(ctes_d[1]*EL*(cos(phi1L)-cos(phi2L))/(ctes_d[2]*ctes_d[4]))*(vz-ctes_d[5]);
	} else {
		result = ctes_d[1]*sqrtf(1-pow(ctes_d[6],2.0))*ctes_d[23]*(cos(phi1zpf)+cos(phi2zpf))/ctes_d[2];
 	}
	return result;
}

__device__ double g(double eta1, double eta2, double t, double y, double z , double vy)
{
	double result=0.0;
	if (z<=ctes_d[25]){
		//double phi1L=ctes_d[16]*t-ctes_d[20]*z-ctes_d[18]*y;
		//double phi2L=ctes_d[16]*t-ctes_d[20]*z+ctes_d[18]*y;
		double phi1L=ctes_d[15]*t-ctes_d[18]*y;
		double phi2L=ctes_d[15]*t+ctes_d[18]*y;
		double EL=ctes_d[22]*exp(-(pow(z-ctes_d[25]/2.0,2.0))/(2.0*pow(ctes_d[25]/5.0,2.0)));
		result = (ctes_d[1]*EL/ctes_d[2])*((cos(phi1L)+cos(phi2L))/ctes_d[7]-(cos(phi1L)-cos(phi2L))*vy/ctes_d[4]);
	}
	return result;
}

// this kernel uses RK4 to calculate the trajectory of a single electron
__global__ void particle_path(double *ang, double *pos, int N) // Without
//__global__ void particle_path(double *ang, double *pos, double *posy, double *posz, int rows, int N) // With
{
	int idx=threadIdx.x+BLOCK_SIZE*blockIdx.x;
	//printf("idx=%d\n",idx);
	//printf("N=%d\n",N);
	if (idx<N){
		double ti=0.0;
		double yi=ang[3*idx+2];
		double zi=0.0;
		double vyi=0.0;
		double vzi=ctes_d[5];
		double yii,zii,vyii,vzii;
		//double kvy1,kvy2,kvy3,kvy4,kvz1,kvz2,kvz3,kvz4;
		
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */		
		//atomicAdd(&(posy[idx*rows]), yi); // Trajectories y
		//atomicAdd(&(posz[idx*rows]), zi); // Trajectories z
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
		//printf("Particle %i has initial position %f with phases eta1=%f and eta2=%f\n",idx,yi,ang[3*idx],ang[3*idx+1]);
		//printf("i=%d\n",i);
		//printf("zi=%f\n",zi);
		//printf("ctes_d[26]=%f\n",ctes_d[26]);
		//printf("rows=%d\n",rows);
		/*printf("");
		printf("");
		printf("");
		printf("");
		printf("");*/

		//while (zi<=ctes_d[26]) {
		while (zi<=ctes_d[25]) {                         // TESTING WITHOUT ZPF (?)
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Using RK4 %%%%%%%%%%%%%%%%%%%%% */
			/*kvy1=f(ang[3*idx],ang[3*idx+1],ti,yi,zi,vzi);
			kvz1=g(ang[3*idx],ang[3*idx+1],ti,yi,zi,vyi);
			kvy2=f(ang[3*idx],ang[3*idx+1],ti+ctes_d[24]/2,yi+ctes_d[24]*(vyi+ctes_d[24]*kvy1/2)/2,zi+ctes_d[24]*(vzi+ctes_d[24]*kvz1/2)/2,vzi+ctes_d[24]*kvz1/2);
			kvz2=g(ang[3*idx],ang[3*idx+1],ti+ctes_d[24]/2,yi+ctes_d[24]*(vyi+ctes_d[24]*kvy1/2)/2,zi+ctes_d[24]*(vzi+ctes_d[24]*kvz1/2)/2,vyi+ctes_d[24]*kvy1/2);
			kvy3=f(ang[3*idx],ang[3*idx+1],ti+ctes_d[24]/2,yi+ctes_d[24]*(vyi+ctes_d[24]*kvy2/2)/2,zi+ctes_d[24]*(vzi+ctes_d[24]*kvz2/2)/2,vzi+ctes_d[24]*kvz2/2);
			kvz3=g(ang[3*idx],ang[3*idx+1],ti+ctes_d[24]/2,yi+ctes_d[24]*(vyi+ctes_d[24]*kvy2/2)/2,zi+ctes_d[24]*(vzi+ctes_d[24]*kvz2/2)/2,vyi+ctes_d[24]*kvy2/2);
			kvy4=f(ang[3*idx],ang[3*idx+1],ti+ctes_d[24],yi+ctes_d[24]*(vyi+ctes_d[24]*kvy3),zi+ctes_d[24]*(vzi+ctes_d[24]*kvz3),vzi+ctes_d[24]*kvz3);
			kvz4=g(ang[3*idx],ang[3*idx+1],ti+ctes_d[24],yi+ctes_d[24]*(vyi+ctes_d[24]*kvy3),zi+ctes_d[24]*(vzi+ctes_d[24]*kvz3),vyi+ctes_d[24]*kvy3);
			// Would it be faster to define ky's and kz's?
			vyii=vyi+ctes_d[24]*(kvy1+2.0*kvy2+2.0*kvy3+kvy4)/6.0;
			vzii=vzi+ctes_d[24]*(kvz1+2.0*kvz2+2.0*kvz3+kvz4)/6.0;*/
			
			//yii=yi+ctes_d[24]*(vyi+2*(vyi+ctes_d[24]*kvy1/2)+2*(vyi+ctes_d[24]*kvy2/2)+(vyi+ctes_d[24]*kvy3))/6;
			//zii=zi+ctes_d[24]*(vzi+2*(vzi+ctes_d[24]*kvz1/2)+2*(vzi+ctes_d[24]*kvz2/2)+(vzi+ctes_d[24]*kvz3))/6;
			
			//yii=yi+ctes_d[24]*vyi+powf(ctes_d[24],2.0)*(kvy1+kvy2+kvy3)/6.0;
			//zii=zi+ctes_d[24]*vzi+powf(ctes_d[24],2.0)*(kvz1+kvz2+kvz3)/6.0;

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Using Euler %%%%%%%%%%%%%%%%%%%%% */
			vyii=vyi+ctes_d[24]*f(ang[3*idx],ang[3*idx+1],ti,yi,zi,vzi);
			vzii=vzi+ctes_d[24]*g(ang[3*idx],ang[3*idx+1],ti,yi,zi,vyi);
			
			yii=yi+ctes_d[24]*vyi;
			zii=zi+ctes_d[24]*vzi;
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */			
			ti=ti+ctes_d[24];
			
			vyi=vyii;
			vzi=vzii;
			yi=yii;
			zi=zii;
			
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
			//atomicAdd(&(posy[idx*rows+i+1]), yi); // Trajectories y
			//atomicAdd(&(posz[idx*rows+i+1]), zi); // Trajectories z
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
		
			__syncthreads(); // Does it go here?
			//printf("Proceso finalizado\n");
		}
		//printf("yf=%f\n",yi);
		//printf("zf=%f\n",zi);
		yi=yi+(ctes_d[26]-ctes_d[25])*vyi/vzi;
		//atomicAdd(&(pos[idx]), yi); // Impact positions
		pos[idx]=yi;
	}
}

// random initialization
__device__ double generate( curandState * globalState, int ind)
{
	//int ind = threadIdx.x;
	curandState localState = globalState[ind];
	double RANDOM = 0.0;
	if (((ind+1)%3)==0)
	{
		RANDOM = (15e-6)*curand_normal( &localState );
		//RANDOM = 1e-4;
	} else {
		RANDOM = 2.0*3.141592*(curand_uniform( &localState )-0.5);
	}
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void setup_angles ( curandState * state, unsigned long seed )
{
	int id = threadIdx.x+BLOCK_SIZE*blockIdx.x;
	curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(double *N, curandState* globalState, int n)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id<3*n)
	{
		double number = generate(globalState, id);
		//printf("%f for %i\n", number,id);
		//atomicAdd(&(N[id]), number);
		N[id]=number;
	}
}

// function called from main fortran program
extern "C" void kernel_wrapper_(double *init, double *pos, int *Np, double *theta, double *phi, double *ki, double *dt, double *D, double *zimp, double *v0, double *wL, double *Delta, int *Nk, double *E0L, double *E0zpf) // Should I provide this variables as pointers or not?
//extern "C" void kernel_wrapper_(double *phi, double *pos, double *posy, double *posz, int *rows, int *Np, double *dt, double *D, double *zimp, double *v0, double *E0) // Should I provide this variables as pointers or not?
{
	ctes[0]=3.1415926535; // PI
	ctes[1]=1.6e-19; // q
	ctes[2]=9.10938356e-31; // m
	ctes[3]=1.0545718e-34; // hbar
	ctes[4]=299792458.0; // c
	ctes[5]=*v0;
	ctes[6]=ctes[5]/ctes[4]; // beta
	ctes[7]=1.0/sqrtf(1.0-pow(ctes[6],2.0)); // gamma
	
	ctes[8]=(ctes[2]*pow(ctes[4],2.0))/ctes[3]; // omega_Compton
	ctes[9]=ctes[7]*ctes[8]; // omega_A
	ctes[10]=ctes[6]*ctes[9]; // omega_B
	ctes[11]=ctes[8]/ctes[4]; // k_Compton
	ctes[12]=ctes[7]*ctes[11]; // k_A
	ctes[13]=ctes[6]*ctes[12]; // k_B
	ctes[14]=2.0*ctes[0]/ctes[11]; // lambda_Compton ~ 2.43e-12 m
	
	//ctes[15]=3.54e15; // omega_Laser
	ctes[15]=*wL;
	ctes[16]=ctes[7]*ctes[15]; // omega_A Laser
	ctes[17]=ctes[6]*ctes[16]; // omega_B Laser
	ctes[18]=ctes[15]/ctes[4]; // k_Laser
	ctes[19]=ctes[7]*ctes[18]; // k_A Laser
	ctes[20]=ctes[6]*ctes[19]; // k_B Laser
	ctes[21]=2.0*ctes[0]/ctes[18]; // lambda_Laser ~ 532 nm
	
	ctes[22]=*E0; // E0 laser
	ctes[23]=*Ezpf; // E0 ZPF
	ctes[24]=*dt; // time step
	ctes[25]=*D; // laser beam waist
	ctes[26]=*zimp; // distance from laser to screen

	ctes[27]=*Delta;
	
	cudaMemcpyToSymbol(ctes_d, ctes, 28 * sizeof(double), 0, cudaMemcpyHostToDevice);

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
	double *init_d, *pos_d;
	double *theta_d, *phi_d, *ki_d; // Without
	double Dkappa;
	//double  *phi_d, *pos_d, *posy_d, *posz_d;   // With
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

	Dkappa=((pow(ctes[15]+ctes[27]/2.0,3.0)-pow(ctes[15]-ctes[27]/2.0,3.0))/(3.0*pow(ctes[4],3.0)))/(Nw-1.0); // <-------------- check this expression

	/*printf("Np=%i\n",*Np);
	printf("numerador=%i\n",3*(*Np));
	printf("denominador=%i\n",BLOCK_SIZE);
	printf("division(double)=%f\n",(double)3*(*Np)/BLOCK_SIZE);
	printf("ceil=%f\n",ceil((double)3*(*Np)/BLOCK_SIZE));
	printf("ceil(int)=%i\n",(int)ceil((double)3*(*Np)/BLOCK_SIZE));*/

	int blocks = (int)ceil((double)(*Nk)/TPB);

	//int blocks = 2;

	/*printf("\nRandom generation\n");
	printf("Blocks per grid: %i\n",blocks); 
	printf("Threads per block: %i\n",N);*/
   // Allocate memory on GPU
	cudaMalloc( (void **)&init_d, sizeof(double) * (*Np) );
	cudaMalloc( (void **)&theta_d, sizeof(double) * (*Nk) );
	cudaMalloc( (void **)&phi_d, sizeof(double) * (*Nk) );
	cudaMalloc( (void **)&ki_d, sizeof(double) * (*Nk) );

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
	//cudaMalloc( (void **)&posy_d, (*rows) * sizeof(double) * (*Np)  ); // Trajectories y
	//cudaMalloc( (void **)&posz_d, (*rows) * sizeof(double) * (*Np) ); // Trajectories z
	cudaMalloc( (void **)&pos_d, sizeof(double) * (*Np) ); // Impact positions
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

///////////////////////////////// RANDOM NUMBERS FOR K-SAMPLING /////////////////////////

	curandState* devStates;
	cudaMalloc ( &devStates, (*Nk)*sizeof( curandState ) );

	// setup seeds
	srand(time(0));
	int seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);

	// theta random generation
	cudaMemcpy(theta_d, theta, sizeof(double) * (*Nk), cudaMemcpyHostToDevice );
	kernel_t<<<blocks,TPB>>> (theta_d, devStates, *Nk);

	// phi random generation
	cudaMemcpy(phi_d, phi, sizeof(double) * (*Nk), cudaMemcpyHostToDevice );
	kernel_p<<<blocks,TPB>>> (phi_d, devStates, *Nk);

	// ki generation
	cudaMemcpy(ki_d, ki, sizeof(double) * (*Nk), cudaMemcpyHostToDevice );
	kernel_k<<<blocks,TPB>>> (ki_d, devStates, *Nk);

//////////////////////////////// RANDOM NUMBERS FOR INITIAL POSITION ///////////////////
	curandState* devStates;
	cudaMalloc ( &devStates, (*Np)*sizeof( curandState ) );

	blocks = (int)ceil((double)(*Np)/TPB);

	// setup seeds
	srand(time(0));
	seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);

	cudaMemcpy(init_d, init, sizeof(double) * (*Np), cudaMemcpyHostToDevice );
	kernel_i<<<blocks,TPB>>> (init_d, devStates, *Np);
	
	// Paths

	/*printf("\nNp=%i\n",*Np);
	printf("numerador=%i\n",*Np);
	printf("denominador=%i\n",BLOCK_SIZE);
	printf("division(double)=%f\n",(double)(*Np)/BLOCK_SIZE);
	printf("ceil=%f\n",ceil((double)(*Np)/BLOCK_SIZE));
	printf("ceil(int)=%i\n",(int)ceil((double)(*Np)/BLOCK_SIZE));*/

	blocks = (int)ceil((double)(*Np)/BLOCK_SIZE);

	/*printf("\nParticle Path\n");
	printf("Blocks per grid: %i\n",blocks); 
	printf("Threads per block: %i\n",N);*/

/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
	//printf("\n%i\n",(*rows) * sizeof(double) * (*Np));
	//cudaMemcpy( posy_d, posy, (*rows) * sizeof(double) * (*Np), cudaMemcpyHostToDevice ); // Trajectories y
	//cudaMemcpy( posz_d, posz, (*rows) * sizeof(double) * (*Np), cudaMemcpyHostToDevice ); // Trajectories z
	cudaMemcpy( pos_d, pos, sizeof(double) * (*Np), cudaMemcpyHostToDevice ); // Impact positions

	particle_path<<<blocks,N>>>(phi_d,pos_d,*Np); // Without
	//particle_path<<<blocks,N>>>(phi_d,pos_d,posy_d,posz_d,*rows,*Np); // With

   // copy vectors back from GPU to CPU
	cudaMemcpy( phi, phi_d, 3*sizeof(double) * (*Np), cudaMemcpyDeviceToHost );
	
	//cudaMemcpy( posy, posy_d, (*rows) * sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Trajectories y
	//cudaMemcpy( posz, posz_d, (*rows) * sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Trajectories z
	cudaMemcpy( pos, pos_d, sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Impact positions
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
	

   // free GPU memory
	cudaFree(phi_d);
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */	
	//cudaFree(posy_d); // Trajectories y
	//cudaFree(posz_d); // Trajectories z
	cudaFree(pos_d); // Impact positions
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */	
	cudaFree(ctes_d);
	return;
}
