#include<stdio.h>
#include<stdlib.h>
#include<string>
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<curand.h>
#include<curand_kernel.h>

#define TPB 128 // Threads per block

// constant memory
__constant__ double ctes_d[21];
/*
ctes[0]  PI
ctes[1]  q
ctes[2]  m
ctes[3]  hbar
ctes[4]  c
ctes[5]  epsilon_0
ctes[6]  v0
ctes[7]  beta
ctes[8]  gamma
ctes[9]  omega_Compton
ctes[10] k_Compton
ctes[11] lambda_Compton
ctes[12] omega_Laser
ctes[13] k_Laser
ctes[14] lambda_Laser
ctes[15] E0 Laser
ctes[16] dt
ctes[17] D
ctes[18] zimp
ctes[19] Delta
ctes[20] V
*/
double ctes[21];

__device__ double f(double theta, double phi, double k, double xi, double eta1, double eta2, double t, double x, double y, double z , double vy, double vz)
{
	double w=k/ctes_d[4];
	double wA=ctes_d[8]*w;
	double wB=ctes_d[7]*wA;
	double kA=ctes_d[8]*k;
	double kB=ctes_d[7]*kA;
	double sintheta=sin(theta);
	double costheta=cos(theta);
	double sinphi=sin(phi);
	double cosphi=cos(phi);
	double sinxi=sin(xi);
	double cosxi=cos(xi);
	double phi1zpf=(wA+wB*costheta)*t-(kA*costheta+kB)*z-k*sintheta*(x*cosphi+y*sinphi)+eta1;
	double phi2zpf=(wA-wB*costheta)*t+(kA*costheta-kB)*z+k*sintheta*(x*cosphi+y*sinphi)+eta2;
	double E0zpf=sqrt(ctes_d[3]*w/(ctes_d[5]*ctes_d[20]));
	
	double result=0.0;

	result=(ctes_d[1]*E0zpf/ctes_d[2])*(cos(phi1zpf)+cos(phi2zpf))*(1-ctes_d[6]*vz/pow(ctes_d[4],2.0))*(costheta*cosphi*cosxi-sinphi*sinxi);
	result=result+(ctes_d[1]*E0zpf/(ctes_d[2]*ctes_d[4]))*(cos(phi1zpf)-cos(phi2zpf))*(sintheta*sinxi*vy/ctes_d[8]-(cosphi*cosxi-costheta*sinphi*sinxi)*(vz-ctes_d[6]));

	return result;
}

__device__ double g(double theta, double phi, double k, double xi, double eta1, double eta2, double t, double x, double y, double z , double vx, double vz)
{
	double w=k/ctes_d[4];
	double wA=ctes_d[8]*w;
	double wB=ctes_d[7]*wA;
	double kA=ctes_d[8]*k;
	double kB=ctes_d[7]*kA;
	double sintheta=sin(theta);
	double costheta=cos(theta);
	double sinphi=sin(phi);
	double cosphi=cos(phi);
	double sinxi=sin(xi);
	double cosxi=cos(xi);
	double phi1zpf=(wA+wB*costheta)*t-(kA*costheta+kB)*z-k*sintheta*(x*cosphi+y*sinphi)+eta1;
	double phi2zpf=(wA-wB*costheta)*t+(kA*costheta-kB)*z+k*sintheta*(x*cosphi+y*sinphi)+eta2;
	double E0zpf=sqrt(ctes_d[3]*w/(ctes_d[5]*ctes_d[20]));
	
	double result=0.0;

	result=(ctes_d[1]*E0zpf/ctes_d[2])*(cos(phi1zpf)+cos(phi2zpf))*(1-ctes_d[6]*vz/pow(ctes_d[4],2.0))*(costheta*sinphi*cosxi+cosphi*sinxi);
	result=result-(ctes_d[1]*E0zpf/(ctes_d[2]*ctes_d[4]))*(cos(phi1zpf)-cos(phi2zpf))*(sintheta*sinxi*vx/ctes_d[8]+(sinphi*cosxi+costheta*cosphi*sinxi)*(vz-ctes_d[6]));

	return result;
}

__device__ double gL(double t, double y, double z , double vz)
{
	double phi1L=ctes_d[12]*t-ctes_d[13]*y;
	double phi2L=ctes_d[12]*t+ctes_d[13]*y;
	double EL=ctes_d[15]*exp(-(pow(z-ctes_d[17]/2.0,2.0))/(2.0*pow(ctes_d[17]/5.0,2.0)));
	
	double result=0.0;

	result = (ctes_d[1]*EL*(cos(phi1L)-cos(phi2L))/(ctes_d[2]*ctes_d[4]))*(vz-ctes_d[6]);
	return result;
}

__device__ double h(double theta, double phi, double k, double xi, double eta1, double eta2, double t, double x, double y, double z, double vx, double vy)
{
	double w=k/ctes_d[4];
	double wA=ctes_d[8]*w;
	double wB=ctes_d[7]*wA;
	double kA=ctes_d[8]*k;
	double kB=ctes_d[7]*kA;
	double sintheta=sin(theta);
	double costheta=cos(theta);
	double sinphi=sin(phi);
	double cosphi=cos(phi);
	double sinxi=sin(xi);
	double cosxi=cos(xi);
	double phi1zpf=(wA+wB*costheta)*t-(kA*costheta+kB)*z-k*sintheta*(x*cosphi+y*sinphi)+eta1;
	double phi2zpf=(wA-wB*costheta)*t+(kA*costheta-kB)*z+k*sintheta*(x*cosphi+y*sinphi)+eta2;
	double E0zpf=sqrt(ctes_d[3]*w/(ctes_d[5]*ctes_d[20]));
	
	double result=0.0;

	result=-(ctes_d[1]*E0zpf/(ctes_d[2]*ctes_d[8]))*(cos(phi1zpf)+cos(phi2zpf))*(sintheta*cosxi);
	result=result+(ctes_d[1]*E0zpf/(ctes_d[2]*ctes_d[4]))*(cos(phi1zpf)-cos(phi2zpf))*((costheta*cosxi-costheta*sinphi*sinxi)*vx+(sinphi*cosxi+costheta*cosphi*sinxi)*vy);

	return result;
}

__device__ double hL(double t, double y, double z , double vy)
{
	double phi1L=ctes_d[12]*t-ctes_d[13]*y;
	double phi2L=ctes_d[12]*t+ctes_d[13]*y;
	double EL=ctes_d[15]*exp(-(pow(z-ctes_d[17]/2.0,2.0))/(2.0*pow(ctes_d[17]/5.0,2.0)));

	double result=0.0;
	
	result = (ctes_d[1]*EL/ctes_d[2])*((cos(phi1L)+cos(phi2L))/ctes_d[8]-(cos(phi1L)-cos(phi2L))*vy/ctes_d[4]);
	return result;
}

// this kernel computes the trajectory of a single electron
__global__ void particle_path(double *theta, double *phi, double *k, double *xhi, double *eta, double *init, double *pos, int N, int Nk) // Without
//__global__ void particle_path(double *ang, double *pos, double *posy, double *posz, int rows, int N) // With
{
	int idx=threadIdx.x+TPB*blockIdx.x;
	//printf("idx=%d\n",idx);
	//printf("N=%d\n",N);
	if (idx<N){
		double ti=0.0;
		double xi=0.0;
		//double yi=init[idx];
		double yi=0.5e-5;
		double zi=0.0;
		double vxi=0.0;
		double vyi=0.0;
		double vzi=ctes_d[6];
		double xii,yii,zii,vxii,vyii,vzii;
		//double kvy1,kvy2,kvy3,kvy4,kvz1,kvz2,kvz3,kvz4;
		
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */		
		//atomicAdd(&(posy[idx*rows]), yi); // Trajectories y
		//atomicAdd(&(posz[idx*rows]), zi); // Trajectories z
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
		//printf("Particle %i has initial position %f with phases eta1=%f and eta2=%f\n",idx,yi,ang[3*idx],ang[3*idx+1]);
		//printf("zi=%f\n",zi);
		//printf("ctes_d[26]=%f\n",ctes_d[26]);
		//printf("rows=%d\n",rows);
		/*printf("");
		printf("");
		printf("");
		printf("");
		printf("");*/

		//while (zi<=ctes_d[26]) {
		while (zi<=ctes_d[17]) {
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
			vxii=f(theta[0],phi[0],k[0],xhi[0],eta[0],eta[1],ti,xi,yi,zi,vxi,vzi);
			for (int j=1;j<Nk;j++)
			{
				vxii=vxii+f(theta[j],phi[j],k[j],xhi[j],eta[2*j],eta[2*j+1],ti,xi,yi,zi,vyi,vzi);
			}
			vxii=vxi+ctes_d[16]*vxii;
			
			vyii=g(theta[0],phi[0],k[0],xhi[0],eta[0],eta[1],ti,xi,yi,zi,vxi,vzi);
			for (int j=1;j<Nk;j++)
			{
				vyii=vyii+g(theta[j],phi[j],k[j],xhi[j],eta[2*j],eta[2*j+1],ti,xi,yi,zi,vxi,vzi);
			}
			vyii=vyii+gL(ti,yi,zi,vzi);
			vyii=vyi+ctes_d[16]*vyii;
			
			vzii=h(theta[0],phi[0],k[0],xhi[0],eta[0],eta[1],ti,xi,yi,zi,vxi,vyi);
			for (int j=1;j<Nk;j++)
			{
				vzii=vzii+h(theta[j],phi[j],k[j],xhi[j],eta[2*j],eta[2*j+1],ti,xi,yi,zi,vxi,vyi);
			}
			vzii=vzii+hL(ti,yi,zi,vyi);
			vzii=vzi+ctes_d[16]*vzii;

			//vyii=vyi+ctes_d[16]*gL(ti,yi,zi,vzi);
			//vzii=vzi+ctes_d[16]*hL(ti,yi,zi,vyi);

			xii=xi+ctes_d[16]*vxi;
			yii=yi+ctes_d[16]*vyi;
			zii=zi+ctes_d[16]*vzi;
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
			ti=ti+ctes_d[16];
			vxi=vxii;
			vyi=vyii;
			vzi=vzii;
			xi=xii;
			yi=yii;
			zi=zii;
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
			__syncthreads(); // Does it go here?
		}
		yi=yi+(ctes_d[18]-ctes_d[17])*vyi/vzi;
		//printf("yi=%f",yi);
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
		RANDOM = 2.0*(curand_uniform( &localState )-0.5);
		//printf("rnd=%lf\n",RANDOM);
		RANDOM = acos(RANDOM);
		//printf("acos(rnd)=%lf\n",RANDOM);
	} else if (i==2)
	{
		//RANDOM = 2.0*3.141592*(curand_uniform( &localState )-0.5);
		RANDOM = 2.0*ctes_d[0]*curand_uniform( &localState );
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
__device__ double genKappa ( double wres, double Dk, int i )
{
	double kappa=0.0;
	kappa=pow(wres-ctes_d[19]/2.0,3.0)/(3.0*(pow(ctes_d[4],3.0)))+(i-1)*Dk;
	return kappa;
}

__global__ void kernel_k ( double *k, double wres, double Dkappa, int n )
{
	int id = threadIdx.x + TPB * blockIdx.x;
	double kappa=genKappa(wres,Dkappa,id);
	k[id]=pow(3.0*kappa,1.0/3.0);
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
	if (id<2*n)
	{
		double number = generate(globalState, id, 2);
		eta[id] = number;
	}
}
// function called from main fortran program
extern "C" void kernel_wrapper_(double *init, double *pos, int *Np, double *theta, double *phi, double *k, double *xi, double *eta, double *dt, double *D, double *zimp, double *v0, double *wL, double *Delta, int *Nk, double *E0L)
//extern "C" void kernel_wrapper_(double *phi, double *pos, double *posy, double *posz, int *rows, int *Np, double *dt, double *D, double *zimp, double *v0, double *E0) // Should I provide this variables as pointers or not?
{
	cudaSetDevice(0);
	ctes[0]=3.1415926535; // PI
	ctes[1]=1.6e-19; // q
	ctes[2]=9.10938356e-31; // m
	//ctes[3]=0.0; // use this to see "classical" results
	ctes[3]=1.0545718e-34; // hbar
	ctes[4]=299792458.0; // c
	ctes[5]=8.854187e-12; // epsilon_0
	ctes[6]=*v0;
	ctes[7]=ctes[6]/ctes[4]; // beta
	ctes[8]=1.0/sqrtf(1.0-pow(ctes[7],2.0)); // gamma
	
	ctes[9]=(ctes[2]*pow(ctes[4],2.0))/ctes[3]; // omega_Compton ~ 7.8e20 rad/s
	ctes[10]=ctes[9]/ctes[4]; // k_Compton
	ctes[11]=2.0*ctes[0]/ctes[10]; // lambda_Compton ~ 2.43e-12 m
	
	ctes[12]=*wL; // omega_Laser=3.54e15 rad/s
	ctes[13]=ctes[12]/ctes[4]; // k_Laser
	ctes[14]=2.0*ctes[0]/ctes[13]; // lambda_Laser ~ 532 nm
	
	ctes[15]=*E0L; // E0 laser
	ctes[16]=*dt; // time step
	ctes[17]=*D; // laser beam waist
	ctes[18]=*zimp; // distance from laser to screen

	ctes[19]=*Delta;
/* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */
	double *init_d, *pos_d;
	double *theta_d, *phi_d, *k_d, *xi_d;
	double *eta_d;
	double Dkappa;
	double wres=ctes[9]; // Resonance around w_Compton
	//double wres=ctes[12]; // Resonance around w_Laser
	Dkappa=(pow(wres+ctes[19]/2.0,3.0)-pow(wres-ctes[19]/2.0,3.0))/(3.0*pow(ctes[4],3.0)); // <-------------- check this expression (Vk)
	printf("Vk=%.10e\n",4.0*ctes[0]*Dkappa);
	ctes[20]=2.0*pow(ctes[0],2.0)*(*Nk)/Dkappa; // V <---------- and this one
	printf("V=%.10e\n",ctes[20]);
	printf("EL_i=%.10e\n",sqrt(ctes[3]*ctes[12]/(ctes[5]*ctes[20])));
	Dkappa=Dkappa/((double)(*Nk)-1.0);
	//printf("Dkappa=%lf\n",Dkappa);
	//double  *phi_d, *pos_d, *posy_d, *posz_d;   // With
	cudaMemcpyToSymbol(ctes_d, ctes, 21 * sizeof(double), 0, cudaMemcpyHostToDevice);
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
	cudaMalloc( (void **)&eta_d, 2*sizeof(double) * (*Np) );
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
	cudaMemcpy(k_d, k, sizeof(double) * (*Nk), cudaMemcpyHostToDevice );
	kernel_k<<<blocks,TPB>>> (k_d, wres, Dkappa, *Nk);
	// xi random generation
	/*srand(time(0));
	seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);*/
	cudaMemcpy(xi_d, xi, sizeof(double) * (*Nk), cudaMemcpyHostToDevice );
	kernel_ang<<<blocks,TPB>>> (xi_d, devStates, *Nk, 2);

//////////////////////////////// RANDOM NUMBERS FOR INITIAL POSITION ///////////////////
	////////////////// RANDOM PHASES
	//setup seeds
	cudaMalloc ( &devStates, (*Np)*sizeof( curandState ) );
	blocks = (int)ceil((double)2*(*Np)/TPB);

	srand(time(0));
	seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);

	cudaMemcpy(eta_d, eta, 2*sizeof(double) * (*Np), cudaMemcpyHostToDevice );
	kernel_eta<<<blocks,TPB>>> (eta_d, devStates, *Np);

	//////////////// RANDOM INITIAL POSITIONS
	// setup seeds
	cudaMalloc ( &devStates, (*Np)*sizeof( curandState ) );
	blocks = (int)ceil((double)(*Np)/TPB);
	
	srand(time(0));
	seed = rand();
	setup_kernels<<<blocks,TPB>>>(devStates,seed);

	cudaMemcpy(init_d, init, sizeof(double) * (*Np), cudaMemcpyHostToDevice );
	kernel_i<<<blocks,TPB>>> (init_d, devStates, *Np);

// copy vectors back from GPU to CPU
	cudaMemcpy( theta, theta_d, sizeof(double) * (*Nk), cudaMemcpyDeviceToHost );
	cudaMemcpy( phi, phi_d, sizeof(double) * (*Nk), cudaMemcpyDeviceToHost );
	cudaMemcpy( k, k_d, sizeof(double) * (*Nk), cudaMemcpyDeviceToHost );
	cudaMemcpy( xi, xi_d, sizeof(double) * (*Nk), cudaMemcpyDeviceToHost );
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
	//printf("%f",theta[0]);

	particle_path<<<blocks,TPB>>>(theta_d,phi_d,k_d,xi_d,eta_d,init_d,pos_d,*Np,*Nk); // Without
	//cudaDeviceSynchronize();
	//particle_path<<<blocks,N>>>(phi_d,pos_d,posy_d,posz_d,*rows,*Np); // With

   // copy vectors back from GPU to CPU
	//cudaMemcpy( posy, posy_d, (*rows) * sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Trajectories y
	//cudaMemcpy( posz, posz_d, (*rows) * sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Trajectories z
	cudaMemcpy( eta, eta_d, 2*sizeof(double) * (*Np), cudaMemcpyDeviceToHost );
	cudaMemcpy( init, init_d, sizeof(double) * (*Np), cudaMemcpyDeviceToHost );
	cudaMemcpy( pos, pos_d, sizeof(double) * (*Np), cudaMemcpyDeviceToHost ); // Impact positions
   
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
