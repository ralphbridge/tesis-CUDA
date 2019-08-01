#include<cuda_runtime.h>
#include<stdio.h>
#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include"math.h"

#define TPB 128

#define N 30 // Number of electrons
#define Nk 20 // Number of k-modes
#define Ne 10 // Number of polarizations per k-mode

__constant__ double pi;
__constant__ double q; // electron charge
__constant__ double m; // electron rest mass
//__constant__ double hbar=0.0; // use this to see "classical" results
__constant__ double hbar;
__constant__ double c; // velocity of light in vacuum
__constant__ double eps0;
__constant__ double v0; // electron velocity before laser region
__constant__ double sigma; // electron beam standard deviation
//__constant__ double beta=v0/c;
//__constant__ double gamma=1.0;//pow(1.0-pow(beta,2.0),-0.5);

__constant__ double wC; // Compton frequency
__constant__ double kC; // kC=wC/c
__constant__ double lamC; // lamC=2pi/kC

__constant__ double wL; // Laser frequency
__constant__ double kL; // kL=wL/c
__constant__ double lamL; // lamL=2pi/kL

__constant__ double E0L; // Laser electric field intensity amplitude
__constant__ double D; // Laser beam waist
__constant__ double zimp; // Screen position (origin set right before laser region)
//__constant__ double sigmaL; // laser region standard deviation

__constant__ double damping; // Damping rate (harmonic oscillator approximation)
__constant__ double Delta; // thickness of the spherical shell in k-space
__constant__ double kmin;
__constant__ double kmax;
__constant__ double V; // Estimated total volume of space

__constant__ double dt; // time step necessary to resolve the electron trajectory

void onHost();
void onDevice(double *k,double *theta,double *phi,double *eta,double *xi,double *init,double *pos);
__global__ void setup_kmodes(curandState *state,unsigned long seed);
__global__ void kmodes(double *x,curandState *state,int option,int n);
__global__ void paths_rk2(double *k,double *theta,double *phi,double *xi,double *eta,double *init,double *pos);
__global__ void paths_rk4(double *k,double *theta,double *phi,double *xi,double *eta,double *init,double *pos);
__device__ void f(double &k,double &theta,double &phi,double &xi,double &eta1,double &eta2,double &t,double &x,double &y,double &z,double &vy,double &vz);
__device__ void g(double &k,double &theta,double &phi,double &xi,double &eta1,double &eta2,double &t,double &x,double &y,double &z,double &vx,double &vz);
__device__ void h(double &k,double &theta,double &phi,double &xi,double &eta1,double &eta2,double &t,double &x,double &y,double &z,double &vx,double &vy);

int main(){
	onHost();
	return 0;
}

void onHost(){
	FILE *k_vec,*posit;

	double *k_h,*theta_h,*phi_h; // Spherical coordinates for each k-mode (Nk in total)
	double *xi_h; // Polarization angles for each k-mode (Ne in total): NOT random
	double *eta_h; // Random phases for the ZPF k-modes (2N in total)
	double *init_h,*pos_h; // Initial and final positions (h indicates host allocation)

	k_h=(double*)malloc(Nk*sizeof(double));
	theta_h=(double*)malloc(Nk*sizeof(double));
	phi_h=(double*)malloc(Nk*sizeof(double));

	xi_h=(double*)malloc(Ne*sizeof(double));

	eta_h=(double*)malloc(2*N*sizeof(double));

	init_h=(double*)malloc(N*sizeof(double));
	pos_h=(double*)malloc(N*sizeof(double));

	onDevice(k_h,theta_h,phi_h,eta_h,xi_h,init_h,pos_h);

	k_vec=fopen("k-vectors.txt","w");
	for(int i=0;i<Nk;i++){
		fprintf(k_vec,"%f,%f,%f\n",k_h[i],theta_h[i],phi_h[i]);
	}
	fclose(k_vec);

	posit=fopen("positions.txt","w");
	for(int i=0;i<N;i++){
		fprintf(posit,"%f,%f\n",init_h[i],pos_h[i]);
	}
	fclose(posit);

	free(k_h);
	free(theta_h);
	free(phi_h);
	free(xi_h);
	free(eta_h);
	free(init_h);
	free(pos_h);
}

void onDevice(double *k_h,double *theta_h,double *phi_h,double *eta_h,double *xi_h,double *init_h,double *pos_h){
	/*const int block_calc=(Nk+TPB-1)/TPB;
	const int blocks=(Nk<block_calc ? 32:block_calc); // Maximum number of resident blocks per SM: 32*/
	unsigned int blocks=(Nk+TPB-1)/TPB;

	double pi_h=3.1415926535;
	double q_h=1.6e-19;
	double m_h=9.10938356e-31;
	double hbar_h=1.0545718e-34;
//	double hbar=0; // uncomment this line to see classical results
	double c_h=299792458.0;
	double eps0_h=8.85e-12;
	double v0_h=1.1e7;
	double fwhm_h=25e-6;
	double sigma_h=fwhm/(2.0*sqrt(2.0*log(2.0)));

	double wC_h=m_h*pow(c_h,2.0)/hbar_h;
	double kC_h=wC_h/c_h;
	double lamC_h=2*pi_h/kC_h;

	double lamL_h=532e-9;
	double kL_h=2*pi_h/lamL_h;
	double wL_h=kL_h*c_h;

	double E0L_h=2.6e8;
	double D_h=125e-6;
	double zimp_h=24e-2+D_h;
//	double sigmaL_h=?; // <---------------- An extra free parameter ?

	double damping_h=6.245835e-24;
	double Delta_h=1e7*damping_h*pow(wL_h,2.0);
	double kmin_h=(wL_h-Delta_h/2.0)/c_h;
	double kmax_h=(wL_h+Delta_h/2.0)/c_h;
	double Vk=4.0*pi_h*(pow(kmax,3.0)-pow(kmin,3.0))/3.0;
	double V_h=pow(2.0*pi_h,3.0)*Nk/Vk;

	double dt_h=pi_h/(10.0*(wL_h+Delta_h/2.0));

	cudaMemcpyToSymbol(pi,&pi_h,sizeof(double));
	cudaMemcpyToSymbol(q,&q_h,sizeof(double));
	cudaMemcpyToSymbol(m,&m_h,sizeof(double));
	cudaMemcpyToSymbol(hbar,&hbar_h,sizeof(double));
	cudaMemcpyToSymbol(c,&c_h,sizeof(double));
	cudaMemcpyToSymbol(eps0,&eps0_h,sizeof(double));
	cudaMemcpyToSymbol(v0,&v0_h,sizeof(double));
	cudaMemcpyToSymbol(sigma,&sigma_h,sizeof(double));

	cudaMemcpyToSymbol(wC,&wC_h,sizeof(double));
	cudaMemcpyToSymbol(kC,&kC_h,sizeof(double));
	cudaMemcpyToSymbol(lamC,&lamC_h,sizeof(double));

	cudaMemcpyToSymbol(lamL,&lamL_h,sizeof(double));
	cudaMemcpyToSymbol(kL,&kL_h,sizeof(double));
	cudaMemcpyToSymbol(wL,&wL_h,sizeof(double));

	cudaMemcpyToSymbol(E0L,&E0L_h,sizeof(double));
	cudaMemcpyToSymbol(D,&D_h,sizeof(double));
	cudaMemcpyToSymbol(zimp,&zimp_h,sizeof(double));
//	cudaMemcpyToSymbol(sigmaL,&sigmaL_h,sizeof(double));

	cudaMemcpyToSymbol(damping,&damping_h,sizeof(double));
	cudaMemcpyToSymbol(Delta,&Delta_h,sizeof(double));
	cudaMemcpyToSymbol(kmin,&kmin_h,sizeof(double));
	cudaMemcpyToSymbol(kmax,&kmax_h,sizeof(double));
	cudaMemcpyToSymbol(V,&V_h,sizeof(double));

	cudaMemcpyToSymbol(dt,&dt_h,sizeof(double));

	double *theta_d,*phi_d,*k_d;
	double *xi_d;
	double *eta_d;
	double *pos_d,*init_d; // Vectors in Device (d indicates device allocation)

	printf("Number of particles: %d\n",N);
	printf("Number of k-modes: %d\n",Nk);
	printf("Number of polarizations: %d\n",Ne);
	printf("Threads per block: %d\n",TPB);
	printf("Number of blocks (k-modes): %d\n",blocks);

	cudaMalloc((void**)&k_d,Nk*sizeof(double));
	cudaMalloc((void**)&theta_d,Nk*sizeof(double));
	cudaMalloc((void**)&phi_d,Nk*sizeof(double));

	cudaMalloc((void**)&eta_d,2*N*sizeof(double));

	cudaMalloc((void**)&xi_d,Ne*sizeof(double));

	cudaMalloc((void**)&init_d,N*sizeof(double));
	cudaMalloc((void**)&pos_d,N*sizeof(double));

	/* Randomly generated k-modes inside the spherical shell */

	curandState *devStates;
        cudaMalloc(&devStates,Nk*sizeof(curandState));

	//k
	srand(time(0));
	int seed=rand(); //Setting up the seeds
	setup_kmodes<<<blocks,TPB>>>(devStates,seed);

	kmodes<<<blocks,TPB>>>(k_d,devStates,1,Nk);

	//theta
	kmodes<<<blocks,TPB>>>(theta_d,devStates,2,Nk);

	//phi
	kmodes<<<blocks,TPB>>>(phi_d,devStates,3,Nk);

	cudaMemcpy(k_h,k_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(theta_h,theta_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(phi_h,phi_d,Nk*sizeof(double),cudaMemcpyDeviceToHost);

	/* Randomly generated phases for the CPC modes */

	curandState *devStates_n;
	cudaMalloc(&devStates_n,2*N*sizeof(curandState));

	blocks=(2*N+TPB-1)/TPB;
	printf("Number of blocks (phases): %d\n",blocks);

	//eta
	srand(time(NULL));
	seed=rand(); //Settin up seeds
	setup_kmodes<<<blocks,TPB>>>(devStates_n,seed);

	kmodes<<<blocks,TPB>>>(eta_d,devStates_n,3,2*N);

	cudaMemcpy(eta_h,eta_d,Ne*sizeof(double),cudaMemcpyDeviceToHost);

	/* Polarization modes allocation (in device memory) */
	for(int i=0;i<Ne;i++){
		xi_h[i]=i*2*pi_h/Ne;
	}

	/* Initial positions */

	blocks=(N+TPB-1)/TPB;
	printf("Number of blocks (paths): %d\n",blocks);

	kmodes<<<blocks,TPB>>>(init_d,devStates_n,4,N);

	paths_rk2<<<blocks,TPB>>>(k_d,theta_d,phi_d,xi_d,eta_d,init_d,pos_d);
	//paths_rk4<<<blocks,TPB>>>(k_d,theta_d,phi_d,xi_d,eta_d,init_d,pos_d);

	cudaMemcpy(pos_h,pos_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(devStates);
	cudaFree(devStates_n);
	cudaFree(k_d);
	cudaFree(theta_d);
	cudaFree(phi_d);
	cudaFree(xi_d);
	cudaFree(eta_d);
	cudaFree(init_d);
	cudaFree(pos_d);
}

__global__ void setup_kmodes(curandState *state,unsigned long seed){
        int idx=threadIdx.x+blockIdx.x*blockDim.x;
        curand_init(seed,idx,0,&state[idx]);
}

__global__ void kmodes(double *vec,curandState *globalState,int opt,int n){
	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	curandState localState=globalState[idx];
	if(idx<n){
		if(opt==1){
			vec[idx]=pow((pow(kmax,3.0)-pow(kmin,3.0))*curand_uniform(&localState)+pow(kmin,3.0),1.0/3.0); // Random radii
		}else if(opt==2){
			vec[idx]=acos(1.0-2.0*curand_uniform(&localState)); // Random polar angles
		}else if(opt==3){
			vec[idx]=2.0*pi*curand_uniform(&localState); // Random azimuthal angles
		}else if(opt==4){
			vec[idx]=sigma*curand_normal(&localState); // Random initial positions
		}
		globalState[idx]=localState; // Update current seed state
	}
}

__global__ void paths_rk2(double *k,double *theta,double *phi,double *xi,double *eta,double *init,double *pos){
	unsigned int idx=threadIdx.x+blockIdx.x*blockDim.x;

	if(idx<N){
		double tn=0.0;
		double xn=0.0;
		double yn=init[idx];
		double zn=0.0;

		double vxn=0.0;
		double vyn=0.0;
		__syncthreads();
		double vzn=v0;

		__shared__ double vxnn[blockDim.x]=0.0;
		__shared__ double vynn[blockDim.x]=0.0;
		__shared__ double vznn[blockDim.x]=0.0;

		__shared__ double k1vx[blockDim.x]=0.0;
		__shared__ double k1vy[blockDim.x]=0.0;
		__shared__ double k1vz[blockDim.x]=0.0;
		__shared__ double k2vx[blockDim.x]=0.0;
		__shared__ double k2vy[blockDim.x]=0.0;
		__shared__ double k2vz[blockDim.x]=0.0;

		while(zn<=D){ // Only laser region. After the particle leaves it, the final position is extrapolated
			for(int i=0;i<Nk;i++){
				for(int j=0;j<Ne;j++){
					f(k1vx[threadIdx.x],k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vyn,vzn); // k1vx represents here the total force in x
					g(k1vy[threadIdx.x],k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn,vzn); // k1vy represents here the total force in y
					h(k1vz[threadIdx.x],k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn,vyn); // k1vz represents here the total force in z
				}
			}

			__syncthreads();
			tn=tn+dt;
			__syncthreads();
			xn=xn+dt*vxn;
			__syncthreads();
			yn=yn+dt*vyn;
			__syncthreads();
			zn=zn+dt*vznn;

			for(int i=0;i<Nk;i++){
				for(int j=0;j<Ne;j++){
					f(k2vx[threadIdx.x],k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vyn+dt*k1vy,vzn+dt*k1vz); // k2vx represents here the total force in x
					g(k2vy[threadIdx.x],k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn+dt*k1vx,vzn+dt*k1vz); // k2vy represents here the total force in y
					h(k2vz[threadIdx.x],k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn+dt*k1vx,vyn+dt*k1vy); // k2vz represents here the total force in z
				}
			}

			__syncthreads();
			vxnn=vxn+dt*(k1vx[threadIdx.x]+k2vx[threadIdx.x])/2;
			__syncthreads();
			vynn=vyn+dt*(k1vy[threadIdx.x]+k2vy[threadIdx.x])/2;
			__syncthreads();
			vznn=vzn+dt*(k1vz[threadIdx.x]+k2vz[threadIdx.x])/2;

			__syncthreads();
			xn=xn+dt*(vxnn[threadIdx.x]-vxn)/2;
			__syncthreads();
			yn=yn+dt*(vynn[threadIdx.x]-vyn)/2;
			__syncthreads();
			zn=zn+dt*(vznn[threadIdx.x]-vzn)/2;

			vxn=vxnn[threadIdx.x];
			vyn=vynn[threadIdx.x];
			vzn=vznn[threadIdx.x];
		}
		__syncthreads();
		pos[idx]=yn+(zimp-D)*vyn/vzn;
	}
}
/*__global__ void paths_rk4(double *k,double *theta,double *phi,double *xi,double *eta,double *init,double *pos){
	unsigned int idx=threadIdx.x+blockIdx.x*blockDim.x;
	__shared__ double tn[blockDim.x];
	__shared__ double xn[blockDim.x];
	__shared__ double yn[blockDim.x];
	__shared__ double zn[blockDim.x];

	__shared__ double vxn[blockDim.x];
	__shared__ double vyn[blockDim.x];
	__shared__ double vzn[blockDim.x];

	__shared__ double vxnn[blockDim.x];
	__shared__ double vynn[blockDim.x];
	__shared__ double vznn[blockDim.x];

	__shared__ double k1vx[blockDim.x];
	__shared__ double k1vy[blockDim.x];
	__shared__ double k1vz[blockDim.x];
	__shared__ double k2vx[blockDim.x];
	__shared__ double k2vy[blockDim.x];
	__shared__ double k2vz[blockDim.x];
	__shared__ double k3vx[blockDim.x];
	__shared__ double k3vy[blockDim.x];
	__shared__ double k3vz[blockDim.x];
	__shared__ double k4vx[blockDim.x];
	__shared__ double k4vy[blockDim.x];
	__shared__ double k4vz[blockDim.x];

	tn[threadIdx.x]=0.0;
	xn[threadIdx.x]=0.0;
	yn[threadIdx.x]=init[idx];
	zn[threadIdx.x]=0.0;

	vxn[threadIdx.x]=0.0;
	vyn[threadIdx.x]=0.0;
	__syncthreads();
	vzn[threadIdx.x]=v0;

	while(zn<=zimp){
		for(int i=0;i<Nk;i++){
			for(int j=0;i<Ne;i++){
				f(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn[threadIdx.x],xn,yn,zn,vyn,vzn); // k1vx represents here the total force in x
		g(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn,vzn); // k1vy represents here the total force in y
		h(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn,vyn); // k1vz represents here the total force in z

		__syncthreads();
		tn=tn+dt/2;
		__syncthreads();
		xn=xn+dt*vxn/2;
		__syncthreads();
		yn=yn+dt*vyn/2;
		__syncthreads();
		zn=zn+dt*vzn/2;
		for(int i=0;i<Nk;i++){
			for(int j=0;j<Ne;j++){
				k2vx[threadIdx.x]=k2vx[threadIdx.x]+f(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vyn+dt*k1vy,vzn+dt*k1vz); // k2vx represents here the total force in x
				k2vy[threadIdx.x]=k2vy[threadIdx.x]+g(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn+dt*k1vx,vzn+dt*k1vz); // k2vy represents here the total force in y
				k2vz[threadIdx.x]=k2vz[threadIdx.x]+h(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn+dt*k1vx,vyn+dt*k1vy); // k2vz represents here the total force in z
			}
		}
		__syncthreads();
		tn=tn+dt;
		__syncthreads();
		xn=xn+dt*vxn;
		__syncthreads();
		yn=yn+dt*vyn;
		__syncthreads();
		zn=zn+dt*vzn;
		for(int i=0;i<Nk;i++){
			for(int j=0;j<Ne;j++){
				k2vx[threadIdx.x]=k2vx[threadIdx.x]+f(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vyn+dt*k1vy,vzn+dt*k1vz); // k2vx represents here the total force in x
				k2vy[threadIdx.x]=k2vy[threadIdx.x]+g(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn+dt*k1vx,vzn+dt*k1vz); // k2vy represents here the total force in y
				k2vz[threadIdx.x]=k2vz[threadIdx.x]+h(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn+dt*k1vx,vyn+dt*k1vy); // k2vz represents here the total force in z
			}
		}
		__syncthreads();
		tn=tn+dt;
		__syncthreads();
		xn=xn+dt*vxn;
		__syncthreads();
		yn=yn+dt*vyn;
		__syncthreads();
		zn=zn+dt*vzn;
		for(int i=0;i<Nk;i++){
			for(int j=0;j<Ne;j++){
				k2vx[threadIdx.x]=k2vx[threadIdx.x]+f(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vyn+dt*k1vy,vzn+dt*k1vz); // k2vx represents here the total force in x
				k2vy[threadIdx.x]=k2vy[threadIdx.x]+g(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn+dt*k1vx,vzn+dt*k1vz); // k2vy represents here the total force in y
				k2vz[threadIdx.x]=k2vz[threadIdx.x]+h(k[i],theta[i],phi[i],xi[j],eta[2*i],eta[2*i+1],tn,xn,yn,zn,vxn+dt*k1vx,vyn+dt*k1vy); // k2vz represents here the total force in z
			}
		}

	}

	__syncthreads();
	vxnn=vxn+dt*(k1vx+k2vx)/2;
	__syncthreads();
	vynn=vyn+dt*(k1vy+k2vy)/2;
	__syncthreads();
	vznn=vzn+dt*(k1vz+k2vz)/2;
}*/

__device__ void f(double &kv,double &k,double &theta,double &phi,double &xi,double &eta1,double &eta2,double &t,double &x,double &y,double &z,double &vy,double &vz){ // ZPF, x-component
	__syncthreads();
	double w=k/c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta1;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta2;

	__syncthreads();
	double E0=sqrt(hbar*w/(eps0*V));
	
	__sincthreads();
	kv+=q*E0*(cos(phi1)+cos(phi2))*(cos(theta)*cos(phi)*cos(xi)-sin(phi)*sin(xi))/m;
	__syncthreads();
	kv+=q*E0*(cos(phi1)-cos(phi2))*(sin(theta)*sin(xi)*vy+(cos(theta)*sin(phi)*sin(xi)-cos(phi)*cos(xi))*vz)/(m*c);
}

__device__ void g(double &kv,double &k,double &theta,double &phi,double &xi,double &eta1,double &eta2,double &t,double &x,double &y,double &z,double &vx,double &vz){ // ZPF, y-component
	__syncthreads();
	double w=k/c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta1;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta2;

	__syncthreads();
	double E0=sqrt(hbar*w/(eps0*V));
	
	__sincthreads();
	kv+=q*E0*(cos(phi1)+cos(phi2))*(cos(theta)*sin(phi)*cos(xi)+cos(phi)*sin(xi))/m;
	__syncthreads();
	kv-=q*E0*(cos(phi1)-cos(phi2))*(sin(theta)*sin(xi)*vx+(cos(theta)*cos(phi)*sin(xi)+sin(phi)*cos(xi))*vz)/(m*c);
}

__device__ void gL(double &t,double &y,double &z,double &vz){ // Laser region, y-component
	__syncthreads();
	double phi1=wL*t-kL*y;
	__syncthreads();
	double phi2=wL*t+kL*y;

	__syncthreads();
	double E0=E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	
	__sincthreads();
	kv+=q*E0*(cos(phi1)-cos(phi2))*vz/(m*c);
}

__device__ void h(double &kv,double &k,double &theta,double &phi,double &xi,double &eta1,double &eta2,double &t,double &x,double &y,double &z,double &vx,double &vy){ // ZPF, z-component
	__syncthreads();
	double w=k/c;

	__syncthreads();
	double phi1=w*t-k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta1;
	__syncthreads();
	double phi2=w*t+k*(sin(theta)*cos(phi)*x+sin(theta)*sin(phi)*y+cos(theta)*z)+eta2;

	__syncthreads();
	double E0=sqrt(hbar*w/(eps0*V));
	
	__sincthreads();
	kv-=q*E0*(cos(phi1)+cos(phi2))*(sin(theta)*cos(xi))/m;
	__syncthreads();
	kv+=q*E0*(cos(phi1)-cos(phi2))*((cos(phi)*cos(xi)-cos(theta)*sin(phi)*sin(xi))*vx+(sin(phi)*cos(xi)+cos(theta)*cos(phi)*sin(xi))*vy)/(m*c);
}

__device__ void hL(double &t,double &y,double &z,double &vy){ // Laser region, z-component
__syncthreads();
	double phi1=wL*t-kL*y;
	__syncthreads();
	double phi2=wL*t+kL*y;

	__syncthreads();
	double E0=E0L*exp(-pow(z-D/2.0,2.0)/(2.0*pow(sigmaL,2.0)));
	
	__syncthreads();
	kv+=q*E0*(cos(phi1)+cos(phi2))*(sin(theta)*cos(xi))/m-q*E0*(cos(phi1)-cos(phi2))*vy/(m*c);
}
