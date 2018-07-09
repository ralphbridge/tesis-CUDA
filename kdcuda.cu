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
__constant__ double ctes_d[29];
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
ctes[28] V
*/
double ctes[29];

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

// this kernel computes the trajectory of a single electron
__global__ void particle_path(double *theta, double *phi, double *k, double *init, double *pos, int N, int Nk) // Without
//__global__ void particle_path(double *ang, double *pos, double *posy, double *posz, int rows, int N) // With
{
	int idx=threadIdx.x+TPB*blockIdx.x;
	//printf("idx=%d\n",idx);
	//printf("N=%d\n",N);
	if (idx<N){
		double ti=0.0;
		double yi=init[idx];
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
			vyii=vyi+ctes_d[24]*f(theta[3*idx],phi[3*idx+1],ti,yi,zi,vzi);
			vzii=vzi+ctes_d[24]*g(theta[3*idx],phi[3*idx+1],ti,yi,zi,vyi);
			
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
__device__ double generate( curandState * globalState, int ind, int i=0 )
{
	//int ind = threadIdx.x;
	curandState localState = globalState[ind];
	double RANDOM = 0.0;
	if (i==1)
	{
		//RANDOM = (15e-6)*curand_normal( &localState );
		//RANDOM = 1e-4;
		RANDOM = cos((2.0*curand_uniform( &localState )-0.5));
	} else if (i==2)
	{
		//RANDOM = 2.0*3.141592*(curand_uniform( &localState )-0.5);
		RANDOM = 2.0*3.141592*curand_uniform( &localState );
	} else
	{
		RANDOM = (15e-6)*curand_normal( &localState );
	}
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void setup_kernels ( curandState * state, unsigned long seed )
{
	int id = threadIdx.x + TPB * blockIdx.x;
	curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel_ang ( double *N, curandState* globalState, int n, int i )
{
	int id = threadIdx.x + TPB * blockIdx.x;
	if (id<n)
	{
		double number = generate(globalState, id, i);
		//printf("%f for %i\n", number,id);
		//atomicAdd(&(N[id]), number);
		N[id] = number;
	}
}
__device__ double genKappa ( double Dk, int i )
{
	double kappa=0.0;
	kappa=pow(ctes_d[15]-ctes_d[27]/2.0,3.0)/(3*(pow(ctes_d[4],3.0)))+(i-1)*Dk;
	return kappa;
}

__global__ void kernel_k ( double *k, double Dkappa, int n )
{
	int id = threadIdx.x + TPB * blockIdx.x;
	double kappa=genKappa(Dkappa,id);
	k[id]=pow(3*kappa,1.0/3.0);
}

__global__ void kernel_i ( double *init, curandState* globalState , int n)
{
	int id = threadIdx.x + TPB * blockIdx.x;
	if (id<n)
	{
		double number = generate(globalState, id);
		init[id] = number;
	}
}

__global__ void kernel_eta ( double *eta, curandState* globalState , int n)
{
	int id = threadIdx.x + TPB * blockIdx.x;
	if (id<n)
	{
		double number = generate(globalState, id, 2);
		eta[id] = number;
	}
}
// function called from main fortran program
extern "C" void kernel_wrapper_(double *init, double *pos, int *Np, double *theta, double *phi, double *k, double *xi, double *eta, double *dt, double *D, double *zimp, double *v0, double *wL, double *Delta, int *Nk, double *E0L, double *E0zpf) // Should I provide this variables as pointers or not?
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
	
	ctes[22]=*E0L; // E0 laser
	ctes[23]=*E0zpf; // E0 ZPF
	ctes[24]=*dt; // time step
	ctes[25]=*D; // laser beam waist
	ctes[26]=*zimp; // distance from laser to screen

	ctes[27]=*Delta;
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
	double *init_d, *pos_d;
	double *theta_d, *phi_d, *k_d, *xi_d;
	double *eta_d;
	double Dkappa;
	Dkappa=(pow(ctes[15]+ctes[27]/2.0,3.0)-pow(ctes[15]-ctes[27]/2.0,3.0))/(3.0*pow(ctes[4],3.0)); // <-------------- check this expression
	ctes[28]=2.0*pow(ctes[0],2.0)*(*Nk)/Dkappa; // V <---------- and this one
	Dkappa=Dkappa/((double)(*Nk)-1.0);
	//printf("Dkappa=%lf\n",Dkappa);
	//double  *phi_d, *pos_d, *posy_d, *posz_d;   // With
	cudaMemcpyToSymbol(ctes_d, ctes, 29 * sizeof(double), 0, cudaMemcpyHostToDevice);
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

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
	cudaMalloc( (void **)&theta_d, sizeof(double) * (*Nk) );
	cudaMalloc( (void **)&phi_d, sizeof(double) * (*Nk) );
	cudaMalloc( (void **)&k_d, sizeof(double) * (*Nk) );
	cudaMalloc( (void **)&xi_d, sizeof(double) * (*Nk) );
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
	//cudaMalloc( (void **)&posy_d, (*rows) * sizeof(double) * (*Np)  ); // Trajectories y
	//cudaMalloc( (void **)&posz_d, (*rows) * sizeof(double) * (*Np) ); // Trajectories z
	cudaMalloc( (void **)&eta_d, sizeof(double) * (*Np) );
	cudaMalloc( (void **)&init_d, sizeof(double) * (*Np) );
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
	kernel_ang<<<blocks,TPB>>> (theta_d, devStates, *Nk, 1);

	// phi random generation
	/*srand(time(0));
	seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);*/
	cudaMemcpy(phi_d, phi, sizeof(double) * (*Nk), cudaMemcpyHostToDevice );
	kernel_ang<<<blocks,TPB>>> (phi_d, devStates, *Nk, 2);

//////////////////////////////// k radii (not random) ///////////////////////////////
	//Dkappa=1.0;
	/*Dkappa=pow(ctes[15]+ctes[27]/2.0,3.0);
	printf("Dkappa=%lf\n",Dkappa);
	Dkappa=-pow(ctes[15]-ctes[27]/2.0,3.0);
	printf("Dkappa=%lf\n",Dkappa);
	Dkappa=(3.0*pow(ctes[4],3.0));
	printf("Dkappa=%lf\n",Dkappa);
	printf("Dkappa=%lf\n",1/((double)(*Nk)-1.0));
	Dkappa=(pow(ctes[15]+ctes[27]/2.0,3.0)-pow(ctes[15]-ctes[27]/2.0,3.0))/(3.0*pow(ctes[4],3.0)); // <-------------- check this expression
	printf("Dkappa=%lf\n",Dkappa);
	Dkappa=Dkappa/((double)(*Nk)-1.0);
	printf("Dkappa=%lf\n",Dkappa);*/

	cudaMemcpy(k_d, k, sizeof(double) * (*Nk), cudaMemcpyHostToDevice );
	kernel_k<<<blocks,TPB>>> (k_d, Dkappa, *Nk);
	// xi random generation
	/*srand(time(0));
	seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);*/
	cudaMemcpy(xi_d, xi, sizeof(double) * (*Nk), cudaMemcpyHostToDevice );
	kernel_ang<<<blocks,TPB>>> (xi_d, devStates, *Nk, 2);

//////////////////////////////// RANDOM NUMBERS FOR INITIAL POSITION ///////////////////
	//curandState* devStates;
	cudaMalloc ( &devStates, (*Np)*sizeof( curandState ) );

	blocks = (int)ceil((double)(*Np)/TPB);

	// setup seeds
	srand(time(0));
	seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);

	cudaMemcpy(init_d, init, sizeof(double) * (*Np), cudaMemcpyHostToDevice );
	kernel_i<<<blocks,TPB>>> (init_d, devStates, *Np);
	
	//setup seeds
	srand(time(0));
	seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);

	cudaMemcpy(eta_d, eta, sizeof(double) * (*Np), cudaMemcpyHostToDevice );
	kernel_eta<<<blocks,TPB>>> (eta_d, devStates, *Np);

// copy vectors back from GPU to CPU
	cudaMemcpy( theta, theta_d, sizeof(double) * (*Nk), cudaMemcpyDeviceToHost );
	cudaMemcpy( phi, phi_d, sizeof(double) * (*Nk), cudaMemcpyDeviceToHost );
	cudaMemcpy( k, k_d, sizeof(double) * (*Nk), cudaMemcpyDeviceToHost );
	cudaMemcpy( xi, xi_d, sizeof(double) * (*Nk), cudaMemcpyDeviceToHost );
	cudaMemcpy( eta, eta_d, sizeof(double) * (*Np), cudaMemcpyDeviceToHost );
////////////////////////////// PATHS ///////////////////////////////////////////

	/*printf("\nNp=%i\n",*Np);
	printf("numerador=%i\n",*Np);
	printf("denominador=%i\n",BLOCK_SIZE);
	printf("division(double)=%f\n",(double)(*Np)/BLOCK_SIZE);
	printf("ceil=%f\n",ceil((double)(*Np)/BLOCK_SIZE));
	printf("ceil(int)=%i\n",(int)ceil((double)(*Np)/BLOCK_SIZE));*/

	/*printf("\nParticle Path\n");
	printf("Blocks per grid: %i\n",blocks); 
	printf("Threads per block: %i\n",N);*/

	//printf("\n%i\n",(*rows) * sizeof(double) * (*Np));
	//cudaMemcpy( posy_d, posy, (*rows) * sizeof(double) * (*Np), cudaMemcpyHostToDevice ); // Trajectories y
	//cudaMemcpy( posz_d, posz, (*rows) * sizeof(double) * (*Np), cudaMemcpyHostToDevice ); // Trajectories z
	cudaMemcpy( pos_d, pos, sizeof(double) * (*Np), cudaMemcpyHostToDevice ); // Impact positions

//2	//particle_path<<<blocks,TPB>>>(theta_d,phi_d,k_d,init_d,pos_d,*Np,*Nk); // Without
	//particle_path<<<blocks,N>>>(phi_d,pos_d,posy_d,posz_d,*rows,*Np); // With

   // copy vectors back from GPU to CPU
	//cudaMemcpy( posy, posy_d, (*rows) * sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Trajectories y
	//cudaMemcpy( posz, posz_d, (*rows) * sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Trajectories z
	cudaMemcpy( init, init_d, sizeof(double) * (*Np), cudaMemcpyDeviceToHost );
//3	cudaMemcpy( pos, pos_d, sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Impact positions
   
   // free GPU memory
	cudaFree(theta_d);
	cudaFree(phi_d);
	cudaFree(k_d);
	cudaFree(xi_d);
	cudaFree(eta_d);
	cudaFree(init_d);
	cudaFree(pos_d); // Impact positions
	cudaFree(ctes_d);
	return;
}
