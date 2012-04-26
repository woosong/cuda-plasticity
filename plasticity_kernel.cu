/* Plasticity.cu
 * This file contains the necessary kernels for execution
 *
 */

#ifndef _PLASTICITY_KERNEL_H_
#define _PLASTICITY_KERNEL_H_

#include <stdio.h>
#include "plasticity.h"

#include <cutil.h>
#include <cufft.h>
#include <cublas.h>
#include <cuComplex.h>

#ifdef PYTHON_COMPATIBILITY_TRANSPOSE_FFT
#include "transpose_kernel.cu"
#endif

#ifdef PYTHON_COMPATIBILITY_FFTW
#include "fftw3.h"
#endif

#define Dx (1./(data_type)N)
#define TILEX 16

static cufftHandle g_planr2c;
static cufftHandle g_planc2r;

#ifdef DYNAMIC_NUCLEATION
data_type maxNucleationTimestep;
data_type *beta0dot;
#endif

__host__ data_type
reduceMax( data_type* u, int size );
__host__ void
updateField( data_type* u, data_type timeStep, data_type *rhs, int size);

__device__ __inline__ data_type minmod3(data_type a, data_type b, data_type c)
{
    if (a>0 && b>0 && c>0) {
        return (a<b) ? ((a<c) ? a : c) : ((c<b) ? c : b);
    } else {
        if (a<0 && b<0 && c<0) {
            return (a>b) ? ((a>c) ? a : c) : ((c>b) ? c : b);
        } else
            return 0;
    }
}

__device__ __inline__ void
findDerivatives( data_type* u, int idx, d_dim_vector x, int coord, data_type *deriv_p, data_type *deriv_m, d_dim_vector L)
{
    volatile d_dim_vector d;
    d.x = (coord==0);
    d.y = (coord==1);
#ifdef DIMENSION3
    d.z = (coord==2);
#endif
    volatile data_type diff_p, diff_m;
    volatile data_type val, val_l, val_r;
    val = locate(u, x, idx);
    val_r = locateop(u, x,+,d, idx);
    diff_p = val_r-val;
    val_l = locateop(u, x,-,d, idx);
    diff_m = val-val_l ;
//diff_p = (*(u+((idx)*L.y+((y+dy+L.y)%L.y))*L.x+((x+dx+L.x)%L.x)))- (*(u+((idx)*L.y+((y+L.y)%L.y))*L.x+((x+L.x)%L.x)));
//diff_m = (*(u+((idx)*L.y+((y+L.y)%L.y))*L.x+((x+L.x)%L.x)))- (*(u+((idx)*L.y+((y-dy+L.y)%L.y))*L.x+((x-dx+L.x)%L.x)));

#ifdef theta
    data_type val_rr, val_ll;
    data_type diff_1, diff_2, diff_3;
    val_rr = locateop(u, x,+2*,d, idx);
    diff_1 = theta*(val_rr + val - 2*val_r);
    diff_2 = theta*(val_r + val_l - 2*val);
    diff_3 = 0.5*(val_rr-val_r-val+val_l);
    *deriv_p = (diff_p-0.5*minmod3(diff_1,diff_2,diff_3)) * (data_type)N;

    val_ll = locateop(u, x,-2*,d, idx);
    diff_1 = theta*(val_ll + val - 2*val_l);
    diff_3 = 0.5*(val_r-val-val_l+val_ll);
    *deriv_m = (diff_m+0.5*minmod3(diff_1,diff_2,diff_3)) * (data_type)N;
#else
    /* LLF */
    *deriv_p = diff_p * (data_type)N;
    *deriv_m = diff_m * (data_type)N;
#endif
}

__device__ const int sigma_index[3][3] = {{0,1,2},{1,3,4},{2,4,5}};

__global__ void
symmetricRHS( data_type* u, data_type* sig, data_type* rhs, data_type* velocity, d_dim_vector L )
{
    volatile int bx = blockIdx.x;     
#ifdef DIMENSION3
    int bz = blockIdx.y/L.y;
    int by = blockIdx.y%L.y;
#else
    int by = blockIdx.y;
#endif
    /* x coordinate is split into threads */
    int tx = threadIdx.x;    
    /* Indices of the array this thread will tackle */
    int i = threadIdx.y;
    int j = threadIdx.z;
    int idx = i*3+j;
    int ix = bx*TILEX + tx;
#ifndef DIMENSION3
    int in_idx = by*L.x + ix;
#else
    int in_idx = (bz*L.y+by)*L.x+ix;
#endif
}

__global__ void
centralHJ( data_type* u, data_type* sig, data_type* rhs, data_type* velocity, d_dim_vector L )
{
    volatile int bx = blockIdx.x;     
#ifdef DIMENSION3
    int bz = blockIdx.y/L.y;
    int by = blockIdx.y%L.y;
#else
    int by = blockIdx.y;
#endif
    /* x coordinate is split into threads */
    int tx = threadIdx.x;    
    /* Indices of the array this thread will tackle */
    int idx = threadIdx.y;
  
    // FIXME Let's see if we can avoid this
    int i = idx/3;
    int j = idx%3;

    int ix = bx*TILEX + tx;
#ifndef DIMENSION3
    int in_idx = by*L.x + ix;
#else
    int in_idx = (bz*L.y+by)*L.x+ix;
#endif
    volatile data_type derivative = 0.;
    volatile data_type ax, ay;
#ifdef DIMENSION3
    volatile data_type az;
#endif

#ifdef DIMENSION3
#define NUM_ELEM 6
#define NUM_ELEM2 8
#else
#define NUM_ELEM 4
#define NUM_ELEM2 4
#endif
    // these have all nine components 
    __shared__ data_type du[NUM_ELEM][NUM_COMP][TILEX];

    // these have only one among nine
    // we only need 6 for 3D, but make 8 for using for rhomod
    __shared__ data_type a[NUM_ELEM2][TILEX];
    
    // Specific to plasticity
    // Use a for rhomod, since it's only used in preparation
    //__shared__ data_type rhomod[4][TILEX];
    __shared__ data_type sigma[NUM_SIG_COMP][TILEX];
    __shared__ data_type v[NUM_ELEM2][3][TILEX];

    d_dim_vector x;
    x.x = ix;
    x.y = by;
#ifdef DIMENSION3
    x.z = bz;
#endif

#ifdef VACANCIES
    if (idx < VACANCY_COMP) {
#endif
    // Determine the derivatives
    findDerivatives(u, idx, x, 0, &du[0][idx][tx], &du[1][idx][tx], L);
    findDerivatives(u, idx, x, 1, &du[2][idx][tx], &du[3][idx][tx], L);
#ifdef DIMENSION3
    findDerivatives(u, idx, x, 2, &du[4][idx][tx], &du[5][idx][tx], L);
#endif
#ifdef VACANCIES
    }
#endif


#if NUM_SIG_COMP > NUM_COMP
#error Number of stress components must be less than or equal to number of components (otherwise modify line below to work)
#endif
    if (idx < NUM_SIG_COMP)
    {
        sigma[idx][tx] = locate(sig, x, idx);
#ifdef VACANCIES
        // take the cost of the vacancies off of the stress
        if (i==j) 
         //*(sigma+in_idx+(i*3+j)*Lsize(L)) -= vacancycost * .... 
          sigma[idx][tx] -= vacancycost * locate(u, x, VACANCY_COMP);
#endif
    }
    __syncthreads();

#if 0
    // DEBUG
    if (idx<4)
        *(rhs+idx*L.y*L.x+in_idx) = du[idx][0][0][tx];
    __syncthreads();
    return;
    // DEBUG
#endif
    // checked up until this part. derivatives are correct.
#if 0
    // DEBUG
    *(rhs+idx*L.y*L.x+in_idx) = sigma[i][j][tx];
    __syncthreads();
    return;
    // DEBUG
#endif
    // checked up until this part. stresses are correct.

    // Prepare for calculation / Plasticity specific
#if NUM_ELEM2 > NUM_COMP
#error Error!
#endif
    if (idx<NUM_ELEM2) {
        // Calculate rhomod
        volatile data_type rhomod = 0.;
        volatile int kx = idx%2;
#ifndef DIMENSION3
        volatile int ky = idx/2+2;
#else
        volatile int ky = (idx%4)/2+2;
        volatile int kz = idx/4+4;
#endif

#define IDX(i,j) (i*3+j)
        // rhomod += ux[i,j]*ux[i,j] when i!=x
        rhomod += du[kx][IDX(1,0)][tx]*du[kx][IDX(1,0)][tx];
        rhomod += du[kx][IDX(1,1)][tx]*du[kx][IDX(1,1)][tx];
        rhomod += du[kx][IDX(1,2)][tx]*du[kx][IDX(1,2)][tx];
        rhomod += du[kx][IDX(2,0)][tx]*du[kx][IDX(2,0)][tx];
        rhomod += du[kx][IDX(2,1)][tx]*du[kx][IDX(2,1)][tx];
        rhomod += du[kx][IDX(2,2)][tx]*du[kx][IDX(2,2)][tx];

        // rhomod += uy[i,j]*uy[i,j] when i!=y
        rhomod += du[ky][IDX(0,0)][tx]*du[ky][IDX(0,0)][tx];
        rhomod += du[ky][IDX(0,1)][tx]*du[ky][IDX(0,1)][tx];
        rhomod += du[ky][IDX(0,2)][tx]*du[ky][IDX(0,2)][tx];
        rhomod += du[ky][IDX(2,0)][tx]*du[ky][IDX(2,0)][tx];
        rhomod += du[ky][IDX(2,1)][tx]*du[ky][IDX(2,1)][tx];
        rhomod += du[ky][IDX(2,2)][tx]*du[ky][IDX(2,2)][tx];

#ifdef DIMENSION3
        // rhomod += uz[z,j]*uz[z,j] when i!=z
        rhomod += du[kz][IDX(0,0)][tx]*du[kz][IDX(0,0)][tx];
        rhomod += du[kz][IDX(0,1)][tx]*du[kz][IDX(0,1)][tx];
        rhomod += du[kz][IDX(0,2)][tx]*du[kz][IDX(0,2)][tx];
        rhomod += du[kz][IDX(1,0)][tx]*du[kz][IDX(1,0)][tx];
        rhomod += du[kz][IDX(1,1)][tx]*du[kz][IDX(1,1)][tx];
        rhomod += du[kz][IDX(1,2)][tx]*du[kz][IDX(1,2)][tx];
#endif

        // rhomod -= 2*ux[y,j]*uy[x,j]
        rhomod -= 2*du[kx][IDX(1,0)][tx]*du[ky][IDX(0,0)][tx];
        rhomod -= 2*du[kx][IDX(1,1)][tx]*du[ky][IDX(0,1)][tx];
        rhomod -= 2*du[kx][IDX(1,2)][tx]*du[ky][IDX(0,2)][tx];
#ifdef DIMENSION3
        // rhomod -= 2*ux[z,j]*uz[x,j]
        rhomod -= 2*du[kx][IDX(2,0)][tx]*du[kz][IDX(0,0)][tx];
        rhomod -= 2*du[kx][IDX(2,1)][tx]*du[kz][IDX(0,1)][tx];
        rhomod -= 2*du[kx][IDX(2,2)][tx]*du[kz][IDX(0,2)][tx];
        // rhomod -= 2*uy[z,j]*uz[y,j]
        rhomod -= 2*du[ky][IDX(2,0)][tx]*du[kz][IDX(1,0)][tx];
        rhomod -= 2*du[ky][IDX(2,1)][tx]*du[kz][IDX(1,1)][tx];
        rhomod -= 2*du[ky][IDX(2,2)][tx]*du[kz][IDX(1,2)][tx];
#endif

        if (rhomod < 0.)
            rhomod = 0.;

        // store in shm
#ifdef VACANCIES
        if (idx < VACANCY_COMP)
#endif
        a[idx][tx] = sqrt(rhomod);
    }
#ifndef DIMENSION3
    else {
        if (idx<8) {
            volatile int tidx = idx-4;
#else
    {
        if (idx<8) {
            volatile int tidx = idx;
#endif
            // Calculate velocity
            volatile data_type vx;
            volatile data_type vy;
            volatile data_type vz;
            volatile int kx = tidx%2;
#ifndef DIMENSION3
            volatile int ky = tidx/2+2;
#else
            volatile int ky = (tidx%4)/2+2;
            volatile int kz = tidx/4+4;
#endif

            // v[x] += uy[x][n]*sigma[y][n]
            // v[x] -= ux[y][n]*sigma[y][n]
            // v[x] -= ux[z][n]*sigma[z][n]
            // these two cancel
            // v[x] += ux[x][n]*sigma[x][n]
            // v[x] -= ux[x][n]*sigma[x][n]
            vx = du[ky][IDX(0,0)][tx]*sigma[IDX(1,0)][tx];
            vx -= du[kx][IDX(1,0)][tx]*sigma[IDX(1,0)][tx];
            vx -= du[kx][IDX(2,0)][tx]*sigma[IDX(2,0)][tx];
            vx += du[ky][IDX(0,1)][tx]*sigma[IDX(1,1)][tx];
            vx -= du[kx][IDX(1,1)][tx]*sigma[IDX(1,1)][tx];
            vx -= du[kx][IDX(2,1)][tx]*sigma[IDX(2,1)][tx];
            vx += du[ky][IDX(0,2)][tx]*sigma[IDX(1,2)][tx];
            vx -= du[kx][IDX(1,2)][tx]*sigma[IDX(1,2)][tx];
            vx -= du[kx][IDX(2,2)][tx]*sigma[IDX(2,2)][tx];
#ifdef DIMENSION3
            // v[x] += uz[x][n]*sigma[z][n]
            vx += du[kz][IDX(0,0)][tx]*sigma[IDX(2,0)][tx];
            vx += du[kz][IDX(0,1)][tx]*sigma[IDX(2,1)][tx];
            vx += du[kz][IDX(0,2)][tx]*sigma[IDX(2,2)][tx];
#endif
            // v[y] += ux[y][n]*sigma[x][n]
            // v[y] -= uy[x][n]*sigma[x][n]
            // v[y] -= uy[z][n]*sigma[z][n]
            vy = du[kx][IDX(1,0)][tx]*sigma[IDX(0,0)][tx];
            vy -= du[ky][IDX(0,0)][tx]*sigma[IDX(0,0)][tx];
            vy -= du[ky][IDX(2,0)][tx]*sigma[IDX(2,0)][tx];
            vy += du[kx][IDX(1,1)][tx]*sigma[IDX(0,1)][tx];
            vy -= du[ky][IDX(0,1)][tx]*sigma[IDX(0,1)][tx];
            vy -= du[ky][IDX(2,1)][tx]*sigma[IDX(2,1)][tx];
            vy += du[kx][IDX(1,2)][tx]*sigma[IDX(0,2)][tx];
            vy -= du[ky][IDX(0,2)][tx]*sigma[IDX(0,2)][tx];
            vy -= du[ky][IDX(2,2)][tx]*sigma[IDX(2,2)][tx];
#ifdef DIMENSION3
            // v[y] += uz[y][n]*sigma[z][n]
            vy += du[kz][IDX(1,0)][tx]*sigma[IDX(2,0)][tx];
            vy += du[kz][IDX(1,1)][tx]*sigma[IDX(2,1)][tx];
            vy += du[kz][IDX(1,2)][tx]*sigma[IDX(2,2)][tx];
#endif

            // v[z] += ux[z][n]*sigma[x][n]
            // v[z] += uy[z][n]*sigma[y][n]
            vz = du[kx][IDX(2,0)][tx]*sigma[IDX(0,0)][tx];
            vz += du[ky][IDX(2,0)][tx]*sigma[IDX(1,0)][tx];
            vz += du[kx][IDX(2,1)][tx]*sigma[IDX(0,1)][tx];
            vz += du[ky][IDX(2,1)][tx]*sigma[IDX(1,1)][tx];
            vz += du[kx][IDX(2,2)][tx]*sigma[IDX(0,2)][tx];
            vz += du[ky][IDX(2,2)][tx]*sigma[IDX(1,2)][tx];
#ifdef DIMENSION3
            // v[z] -= uz[x][n]*sigma[x][n]
            // v[z] -= uz[y][n]*sigma[y][n]
            // these two cancel
            // v[z] += uz[z][n]*sigma[z][n]
            // v[z] -= uz[z][n]*sigma[z][n]
            vz -= du[kz][IDX(0,0)][tx]*sigma[IDX(0,0)][tx];
            vz -= du[kz][IDX(0,1)][tx]*sigma[IDX(0,1)][tx];
            vz -= du[kz][IDX(0,2)][tx]*sigma[IDX(0,2)][tx];
            vz -= du[kz][IDX(1,0)][tx]*sigma[IDX(2,0)][tx];
            vz -= du[kz][IDX(1,1)][tx]*sigma[IDX(2,1)][tx];
            vz -= du[kz][IDX(1,2)][tx]*sigma[IDX(2,2)][tx];
#endif

#ifndef SLIPSYSTEMS
            // FIXME - glide term
            data_type sigma_tr = (sigma[IDX(0,0)][tx]+sigma[IDX(1,1)][tx]+sigma[IDX(2,2)][tx])/3.*lambda;
            vx += sigma_tr*(du[kx][IDX(0,0)][tx]+du[kx][IDX(1,1)][tx]+du[kx][IDX(2,2)][tx]);
            vy += sigma_tr*(du[ky][IDX(0,0)][tx]+du[ky][IDX(1,1)][tx]+du[ky][IDX(2,2)][tx]);
            vx -= sigma_tr*(du[kx][IDX(0,0)][tx]+du[ky][IDX(0,1)][tx]);
            vy -= sigma_tr*(du[kx][IDX(1,0)][tx]+du[ky][IDX(1,1)][tx]);
            vz -= sigma_tr*(du[kx][IDX(2,0)][tx]+du[ky][IDX(2,1)][tx]);
#ifdef DIMENSION3
            vz += sigma_tr*(du[kz][IDX(0,0)][tx]+du[kz][IDX(1,1)][tx]+du[kz][IDX(2,2)][tx]);
            vx -= sigma_tr*(du[kz][IDX(0,2)][tx]);
            vy -= sigma_tr*(du[kz][IDX(1,2)][tx]);
            vz -= sigma_tr*(du[kz][IDX(2,2)][tx]);
#endif
#endif


#ifdef NEWGLIDEONLY
#ifndef DIMENSION3
	volatile data_type rhorho = ME;
 	volatile data_type uxTr   = 0.;
	volatile data_type uyTr   = 0.;

	uxTr = du[kx][IDX(0,0)][tx]+du[kx][IDX(1,1)][tx]+du[kx][IDX(2,2)][tx];
	uyTr = du[ky][IDX(0,0)][tx]+du[ky][IDX(1,1)][tx]+du[ky][IDX(2,2)][tx];

	rhorho += uxTr*uxTr + uyTr*uyTr;
	rhorho += (du[kx][IDX(0,0)][tx]+du[ky][IDX(0,1)][tx])*
		      (du[kx][IDX(0,0)][tx]+du[ky][IDX(0,1)][tx]);
	rhorho += (du[kx][IDX(1,0)][tx]+du[ky][IDX(1,1)][tx])*
		      (du[kx][IDX(1,0)][tx]+du[ky][IDX(1,1)][tx]);
	rhorho += (du[kx][IDX(2,0)][tx]+du[ky][IDX(2,1)][tx])*
		      (du[kx][IDX(2,0)][tx]+du[ky][IDX(2,1)][tx]);
	rhorho -= 2*uxTr*(du[kx][IDX(0,0)][tx] + du[ky][IDX(0,1)][tx]);
	rhorho -= 2*uyTr*(du[kx][IDX(1,0)][tx] + du[ky][IDX(1,1)][tx]);
        
	volatile data_type rhov = ME;

	rhov += vx*(du[kx][IDX(2,2)][tx]+du[kx][IDX(1,1)][tx]-du[ky][IDX(0,1)][tx]);
	rhov += vy*(du[ky][IDX(0,0)][tx]+du[ky][IDX(2,2)][tx]-du[kx][IDX(1,0)][tx]);
	rhov -= vz*(du[kx][IDX(2,0)][tx]+du[ky][IDX(2,1)][tx]);

    rhov = rhov/rhorho;
	
	vx += (du[kx][IDX(1,1)][tx]+du[kx][IDX(2,2)][tx]-du[ky][IDX(0,1)][tx])*rhov;
	vy += (du[ky][IDX(0,0)][tx]+du[ky][IDX(2,2)][tx]-du[kx][IDX(1,0)][tx])*rhov;
	vz -= (du[kx][IDX(2,0)][tx]+du[ky][IDX(2,1)][tx])*rhov;
#else
	volatile data_type rhorho = ME;
    volatile data_type uxTr   = 0.;
    volatile data_type uyTr   = 0.;
	volatile data_type uzTr   = 0.;

    uxTr = du[kx][IDX(0,0)][tx]+du[kx][IDX(1,1)][tx]+du[kx][IDX(2,2)][tx];
    uyTr = du[ky][IDX(0,0)][tx]+du[ky][IDX(1,1)][tx]+du[ky][IDX(2,2)][tx];
	uzTr = du[kz][IDX(0,0)][tx]+du[kz][IDX(1,1)][tx]+du[kz][IDX(2,2)][tx];

    rhorho += uxTr*uxTr + uyTr*uyTr + uzTr*uzTr;
    rhorho += (du[kx][IDX(0,0)][tx]+du[ky][IDX(0,1)][tx]+du[kz][IDX(0,2)][tx])*
              (du[kx][IDX(0,0)][tx]+du[ky][IDX(0,1)][tx]+du[kz][IDX(0,2)][tx]);
    rhorho += (du[kx][IDX(1,0)][tx]+du[ky][IDX(1,1)][tx]+du[kz][IDX(1,2)][tx])*
              (du[kx][IDX(1,0)][tx]+du[ky][IDX(1,1)][tx]+du[kz][IDX(1,2)][tx]);
    rhorho += (du[kx][IDX(2,0)][tx]+du[ky][IDX(2,1)][tx]+du[kz][IDX(2,2)][tx])*
              (du[kx][IDX(2,0)][tx]+du[ky][IDX(2,1)][tx]+du[kz][IDX(2,2)][tx]);
    rhorho -= 2*uxTr*(du[kx][IDX(0,0)][tx] + du[ky][IDX(0,1)][tx] + du[kz][IDX(0,2)][tx]);
    rhorho -= 2*uyTr*(du[kx][IDX(1,0)][tx] + du[ky][IDX(1,1)][tx] + du[kz][IDX(1,2)][tx]);
	rhorho -= 2*uzTr*(du[kx][IDX(2,0)][tx] + du[ky][IDX(2,1)][tx] + du[kz][IDX(2,2)][tx]);

    volatile data_type rhov = ME;

    rhov += vx*(du[kx][IDX(2,2)][tx]+du[kx][IDX(1,1)][tx]-du[ky][IDX(0,1)][tx]-du[kz][IDX(0,2)][tx]);
    rhov += vy*(du[ky][IDX(0,0)][tx]+du[ky][IDX(2,2)][tx]-du[kx][IDX(1,0)][tx]-du[kz][IDX(1,2)][tx]);
    rhov += vz*(du[kz][IDX(0,0)][tx]+du[kz][IDX(1,1)][tx]-du[kx][IDX(2,0)][tx]-du[ky][IDX(2,1)][tx]);

    rhov = rhov/rhorho;

    vx += (du[kx][IDX(1,1)][tx]+du[kx][IDX(2,2)][tx]-du[ky][IDX(0,1)][tx]-du[kz][IDX(0,2)][tx])*rhov;
    vy += (du[ky][IDX(0,0)][tx]+du[ky][IDX(2,2)][tx]-du[kx][IDX(1,0)][tx]-du[kz][IDX(1,2)][tx])*rhov;
    vz += (du[kz][IDX(0,0)][tx]+du[kz][IDX(1,1)][tx]-du[kx][IDX(2,0)][tx]-du[ky][IDX(2,1)][tx])*rhov;
#endif
#endif
             // store in shm
            v[tidx][0][tx] = vx;
            v[tidx][1][tx] = vy;
            v[tidx][2][tx] = vz;
#ifdef DIMENSION3
        }
    }
#else
        }
    }
#endif
    __syncthreads();

    // FIXME - checked up until this part. rho and velocities are correct.

    if (idx < NUM_ELEM2) {
        volatile data_type irhomod = a[idx][tx];
#ifndef PYTHON_COMPATIBILITY_DIVIDE
#ifndef SLOPPY_NO_DIVIDE_BY_ZERO
        if (irhomod == 0.) {
            irhomod = 0.;
        } else {
            irhomod = 1./irhomod;
        } 
#else
        irhomod = 1./(irhomod+ME);
#endif
        // This gives different values - FIXME Check why
        v[idx][0][tx] *= irhomod;
        v[idx][1][tx] *= irhomod;
        v[idx][2][tx] *= irhomod;
        a[idx][tx] = 0.;
#else
        irhomod += sqrt(ME);
        v[idx][0][tx] /= irhomod;
        v[idx][1][tx] /= irhomod;
        v[idx][2][tx] /= irhomod;
        a[idx][tx] = 0.;
#endif
    }
    __syncthreads();
#if 0
    // DEBUG
    if (idx<4)
        *(rhs+(idx)*L.y*L.x+in_idx) = v[idx][0][tx];
    if (idx>=4 && idx<8)
        *(rhs+(idx)*L.y*L.x+in_idx) = v[idx-4][1][tx];
    __syncthreads();
    return;
    // DEBUG
#endif

    // Calculate the velocities 
    // NOTE: diverging branches for different i&j are not problems! 
    // because they are different warps (or the hope is so)
    // however one needs to be careful that they do not need to excute in order.
#ifdef LLF
    if (idx==0) {
        volatile data_type max = 0.;
#ifndef SLIPSYSTEMS
        for(int k = 0; k<NUM_ELEM2; k++) {
            volatile data_type t = v[k][idx][tx];
            if (max < t) 
                max = t;
            if (max < -t)
                max = -t;
        }
#else
        //slip systems need different velocities and LLF only implemented
        for(int k = 0; k<3; k++) {
            for(int l = 0; l<3; l++) {
                max += sigma[IDX(k,l)][tx]*sigma[IDX(k,l)][tx];
            }
        }
        max = sqrt(max);
#endif
        a[0][tx] = max;
        a[1][tx] = max;
        a[2][tx] = max;
        a[3][tx] = max;
#ifdef DIMENSION3
        a[4][tx] = max;
        a[5][tx] = max;
#endif
    }
#else
#ifdef SLIPSYSTEMS
#error No support for non-LLF for slip systems yet
#endif
#ifndef DIMENSION3
    if (idx<2) 
#else
    if (idx<3)
#endif
    {
        // Calculate ab_pm
        for(int k = 0; k<NUM_ELEM2; k++) {
            volatile data_type t = v[k][idx][tx];
            if (a[idx*2][tx] < t) 
                a[idx*2][tx] = t;
            if (a[idx*2+1][tx] < -t)
                a[idx*2+1][tx] = -t;
        }
    } 
#endif
    __syncthreads();
#if 0
    // DEBUG
    if (idx<4)
        *(rhs+(idx)*L.y*L.x+in_idx) = a[idx][tx];
    if (idx>=4 && idx<8)
        *(rhs+(idx)*L.y*L.x+in_idx) = v[idx-4][0][tx];
    __syncthreads();
    return;
    // DEBUG
#endif

    // FIXME do this more elegantly
    ax = a[0][tx]+a[1][tx]+ME;
    ay = a[2][tx]+a[3][tx]+ME;
#ifdef DIMENSION3
    az = a[4][tx]+a[5][tx]+ME;
#endif

    // Calculate hamiltonian and derivative using these results
#ifdef VACANCIES
    if (idx < VACANCY_COMP)
    {
#endif
    for(int k=0; k<NUM_ELEM2; k++) {
        volatile data_type h;
        volatile int kx = k%2;
#ifndef DIMENSION3
        volatile int ky = k/2+2;
#else
        volatile int ky = (k%4)/2+2;
        volatile int kz = k/4+4;
#endif

#ifndef SLIPSYSTEMS
        // H_ij = v_l (d_l beta_ij - d_i beta_lj)
        h = du[kx][idx][tx] * v[k][0][tx] + du[ky][idx][tx] * v[k][1][tx];
#ifdef DIMENSION3
        h += du[kz][idx][tx] * v[k][2][tx];
#endif
        if (i==0) {
            h -= v[k][0][tx] * du[kx][IDX(0,j)][tx];
            h -= v[k][1][tx] * du[kx][IDX(1,j)][tx];
            h -= v[k][2][tx] * du[kx][IDX(2,j)][tx];
        } else {
            if (i==1) {
                h -= v[k][0][tx] * du[ky][IDX(0,j)][tx];
                h -= v[k][1][tx] * du[ky][IDX(1,j)][tx];
                h -= v[k][2][tx] * du[ky][IDX(2,j)][tx];
            }
#ifdef DIMENSION3
            else {
                if (i==2) {
                    h -= v[k][0][tx] * du[kz][IDX(0,j)][tx];
                    h -= v[k][1][tx] * du[kz][IDX(1,j)][tx];
                    h -= v[k][2][tx] * du[kz][IDX(2,j)][tx];
                }
            }
#endif
        }
        // Checked up until this point
        // Glide only correction
        // H_ij += lambda v_l / 3 (d_l beta_kk - d_k beta_lk)
        if (i==j) {
            h -= (v[k][0][tx]*(du[kx][IDX(1,1)][tx]+du[kx][IDX(2,2)][tx]-du[ky][IDX(0,1)][tx])
                    +v[k][1][tx]*(du[ky][IDX(0,0)][tx]+du[ky][IDX(2,2)][tx]-du[kx][IDX(1,0)][tx])
                    -v[k][2][tx]*(du[kx][IDX(2,0)][tx]+du[ky][IDX(2,1)][tx])
#ifdef DIMENSION3
                    -v[k][0][tx]*(du[kz][IDX(0,2)][tx])
                    -v[k][1][tx]*(du[kz][IDX(1,2)][tx])
                    +v[k][2][tx]*(du[kz][IDX(0,0)][tx]+du[kz][IDX(1,1)][tx]) 
#endif
                )/3.*lambda;
        }   
#else
        // Slip systems dynamics
        // H_ij = sigma_ij (|D beta_ij|)
        volatile data_type rho = 0.;
        if (i!=0) rho += abs(du[kx][idx][tx]);
        if (i!=1) rho += abs(du[ky][idx][tx]);
#ifdef DIMENSION3
        if (i!=2) rho += abs(du[kz][idx][tx]);
#endif
        // FIXME - need to be careful when extending fields
        h = sigma[idx][tx]*rho;

        // H_ij^prime = (1-m)H_ij + m \sum_{k!=i} V_k d_k beta_ij 
#ifdef mixing
        h *= (1.-mixing);
        if (i!=0) h += mixing*v[k][0][tx]*(du[kx][idx][tx]);
        if (i!=1) h += mixing*v[k][1][tx]*(du[ky][idx][tx]);
#ifdef DIMENSION3
        if (i!=2) h += mixing*v[k][2][tx]*(du[kz][idx][tx]);
#endif
#endif //endif mixing
#endif //endif SLIPSYSTEMS

        // add all to the derivative
        volatile int nkx = 1-kx;
        volatile int nky = 5-ky;
#ifndef DIMENSION3
        derivative += -h*a[nkx][tx]*a[nky][tx]/ax/ay;
#else
        volatile int nkz = 9-kz;
        derivative += -h*a[nkx][tx]*a[nky][tx]*a[nkz][tx]/ax/ay/az;
#endif
    }

    // diffusion term FIXME
    //if (i!=0)
        derivative += (du[0][idx][tx]-du[1][idx][tx])*a[0][tx]*a[1][tx]/ax;
    //if (i!=1)
        derivative += (du[2][idx][tx]-du[3][idx][tx])*a[2][tx]*a[3][tx]/ay;
#ifdef DIMENSION3
        derivative += (du[4][idx][tx]-du[5][idx][tx])*a[4][tx]*a[5][tx]/az;
#endif
#ifdef VACANCIES
    } // matches idx < VACANCY_COMP
#endif

    *(rhs+idx*Lsize(L)+in_idx) = derivative;

#ifdef VACANCIES
    __syncthreads();
    if (idx == VACANCY_COMP){
        *(rhs+idx*Lsize(L)+in_idx) =  ( *(rhs+0*Lsize(L)+in_idx) +
                                        *(rhs+4*Lsize(L)+in_idx) +
                                        *(rhs+8*Lsize(L)+in_idx)  );
    }
#endif

    if (idx==0)
        *(velocity+in_idx) = ax+ay
#ifdef DIMENSION3
                            +az
#endif
                             ;
}

#ifdef VACANCIES
__global__ void
calculateKDiffusion( cdata_type* Ku, d_dim_vector L, data_type dt )
{
    // This kernel calculates the sigma in k-space
    // Since K-space field is supposedly unnecessary afterwards,
    // it is overwritten
    int bx = blockIdx.x;     
#ifdef DIMENSION3
    int by = blockIdx.y%L.y;
    int bz = blockIdx.y/L.y;
#else
    int by = blockIdx.y;
#endif
    /* x coordinate is split into threads */
    int tx = threadIdx.x;    
    /* Indices of the array this thread will tackle */
    int i = threadIdx.y;
    int j = threadIdx.z;
    int ix = bx*TILEX + tx;
#ifndef DIMENSION3
    int in_idx = by*L.x + ix;
#else
    int in_idx = (bz*L.y+by)*L.x+ix;
#endif

    // FIXME - k values need to be properly dealt with
    data_type kx = ix*2.*M_PI;
    data_type ky = ((by>N/2)?by-N:by)*2.*M_PI;
    data_type kSq = kx*kx + ky*ky;
#ifdef DIMENSION3
    data_type kz = ((bz>N/2)?bz-N:bz)*2.*M_PI;
    kSq += kz*kz;
#else
    data_type kz = 0.;
#endif

    __shared__ cdata_type Ku_shm[TILEX];
 
    if (ix < L.x) {
        Ku_shm[tx] = *(Ku+in_idx);
        volatile data_type diffuse = exp(-vacancydiffusion*kSq*dt);
        Ku_shm[tx].x *= diffuse;
        Ku_shm[tx].y *= diffuse;
#ifdef DIMENSION3
        Ku_shm[tx].x /= N*N*N;
        Ku_shm[tx].y /= N*N*N;
#else
        Ku_shm[tx].x /= N*N;
        Ku_shm[tx].y /= N*N;
#endif
        *(Ku+in_idx) = Ku_shm[tx];
        // Division for Normalization
    }
    __syncthreads();
}

__host__ void
calculateDiffusion( data_type* u, d_dim_vector L, data_type dt )
{
    dim3 grid(KGridSize(L));
    dim3 tids(TILEX, 1, 1);
    cdata_type *Ku;
    CUDA_SAFE_CALL(cudaMalloc((void**) &Ku, sizeof(cdata_type)*LKsize(L)));
    
    // Fourier transform u
    fft_r2c(g_planr2c, (fft_dtype_r*)(u+Lsize(L)*VACANCY_COMP), (fft_dtype_c*)(Ku));
    
    // calculateKSigma
    d_dim_vector newL = L;
    newL.x = L.x/2+1;
    calculateKDiffusion<<<grid, tids>>>(Ku, newL, dt);
    
    // inverse Fourier kSigma
    fft_c2r(g_planc2r, (fft_dtype_c*)(Ku), (fft_dtype_r*)(u+Lsize(L)*VACANCY_COMP));
    CUDA_SAFE_CALL(cudaFree(Ku));
}
#endif

__global__ void
calculateKSigma( cdata_type* Ku, d_dim_vector L )
{
    // This kernel calculates the sigma in k-space
    // Since K-space field is supposedly unnecessary afterwards,
    // it is overwritten
    int bx = blockIdx.x;     
#ifdef DIMENSION3
    int by = blockIdx.y%L.y;
    int bz = blockIdx.y/L.y;
#else
    int by = blockIdx.y;
#endif
    /* x coordinate is split into threads */
    int tx = threadIdx.x;    
    /* Indices of the array this thread will tackle */
    int i = threadIdx.y;
    int j = threadIdx.z;
    //int idx = i*3+j;
    int ix = bx*TILEX + tx;
#ifndef DIMENSION3
    int in_idx = by*L.x + ix;
#else
    int in_idx = (bz*L.y+by)*L.x+ix;
#endif

    // FIXME - k values need to be properly dealt with
#ifndef PYTHON_COMPATIBILITY_TRANSPOSE_FFT
    data_type kx = ix*2.*M_PI;
#else
    data_type kx = ((ix>N/2)?ix-N:ix)*2.*M_PI;
#endif
    data_type ky = ((by>N/2)?by-N:by)*2.*M_PI;
    data_type kSq = kx*kx + ky*ky;
#ifdef DIMENSION3
    data_type kz = ((bz>N/2)?bz-N:bz)*2.*M_PI;
    kSq += kz*kz;
#else
    data_type kz = 0.;
#endif
    data_type kSqSq = kSq*kSq;
    data_type k[3];

#ifdef PYTHON_COMPATIBILITY
    kSq += ME;
    kSqSq += ME;
#endif

    __shared__ cdata_type Ku_shm[3][3][TILEX];
    __shared__ cdata_type Ksig_shm[3][3][TILEX];
    volatile data_type k_i, k_j;
 
    if (ix < L.x) {
        k[0] = kx;
        k[1] = ky;
#ifndef DIMENSION3
        k[2] = 0;
#else
        k[2] = kz;
#endif
        if (j==0)
            k_j = kx;
        if (j==1)
            k_j = ky;
        if (j==2)
            k_j = kz;
        if (i==0)
            k_i = kx;
        if (i==1)
            k_i = ky;
        if (i==2)
            k_i = kz;
        Ku_shm[i][j][tx] = *(Ku+in_idx+Lsize(L)*(i*3+j));
        Ksig_shm[i][j][tx] = init_cdata(0.,0.);
    }
    __syncthreads();
    if (ix < L.x) {
#ifndef DIMENSION3
        if (ix == 0 && by == 0) 
#else
        if (ix == 0 && by == 0 && bz == 0)
#endif
        {
            // This is the constant part
            Ksig_shm[i][j][tx].x = 0.;
            Ksig_shm[i][j][tx].y = 0.;
            // New Boundary condition d_i u_j = 0
#if 1
            if (i==j) {
                cdata_type betaE_trace;
                betaE_trace.x = (Ku_shm[0][0][tx].x+Ku_shm[1][1][tx].x+Ku_shm[2][2][tx].x);
                betaE_trace.y = (Ku_shm[0][0][tx].y+Ku_shm[1][1][tx].y+Ku_shm[2][2][tx].y);
                Ksig_shm[i][j][tx].x -= 2*mu*nu/(1-2*nu)*betaE_trace.x;
                Ksig_shm[i][j][tx].y -= 2*mu*nu/(1-2*nu)*betaE_trace.y;
            }
            Ksig_shm[i][j][tx].x -= mu*(Ku_shm[i][j][tx].x+Ku_shm[j][i][tx].x);
            Ksig_shm[i][j][tx].y -= mu*(Ku_shm[i][j][tx].y+Ku_shm[j][i][tx].y);
#endif
        } else {
            cdata_type temp = init_cdata(0.,0.);
#if 0
#pragma unroll 1
            for(int m = 0; m < 3; m++) {
#pragma unroll 1
                for(int n = 0; n < 3; n++) {
                    data_type M = ((2*mu*nu/(1-nu))*((k[m]*k[n]*(i==j)+k[i]*k[j]*(m==n))/kSq - (i==j)*(m==n)) - mu*((i==m)*(j==n)+(i==n)*(j==m)) - (2*mu/(1-nu))*k[i]*k[j]*k[m]*k[n]/kSqSq + mu*(k[i]*k[n]*(j==m)+k[i]*k[m]*(j==n)+k[j]*k[n]*(i==m)+k[j]*k[m]*(i==n))/kSq);
                    //Ksig_shm[i][j][tx].x += M * Ku_shm[m][n][tx].x;
                    //Ksig_shm[i][j][tx].y += M * Ku_shm[m][n][tx].y;
                    temp.x += M * Ku_shm[m][n][tx].x;
                    temp.y += M * Ku_shm[m][n][tx].y;
                }
            }
#else
                    data_type M = ((2*mu*nu/(1-nu))*((kx*kx*(i==j)+k_i*k_j*(0==0))/kSq - (i==j)*(0==0)) - mu*((i==0)*(j==0)+(i==0)*(j==0)) - (2*mu/(1-nu))*k_i*k_j*kx*kx/kSqSq + mu*(k_i*kx*(j==0)+k_i*kx*(j==0)+k_j*kx*(i==0)+k_j*kx*(i==0))/kSq);
                    temp.x += M * Ku_shm[0][0][tx].x; temp.y += M * Ku_shm[0][0][tx].y;
                    M = ((2*mu*nu/(1-nu))*((kx*ky*(i==j)+k_i*k_j*(0==1))/kSq - (i==j)*(0==1)) - mu*((i==0)*(j==1)+(i==1)*(j==0)) - (2*mu/(1-nu))*k_i*k_j*kx*ky/kSqSq + mu*(k_i*ky*(j==0)+k_i*kx*(j==1)+k_j*ky*(i==0)+k_j*kx*(i==1))/kSq);
                    temp.x += M * Ku_shm[0][1][tx].x; temp.y += M * Ku_shm[0][1][tx].y;
                    M = ((2*mu*nu/(1-nu))*((kx*kz*(i==j)+k_i*k_j*(0==2))/kSq - (i==j)*(0==2)) - mu*((i==0)*(j==2)+(i==2)*(j==0)) - (2*mu/(1-nu))*k_i*k_j*kx*kz/kSqSq + mu*(k_i*kz*(j==0)+k_i*kx*(j==2)+k_j*kz*(i==0)+k_j*kx*(i==2))/kSq);
                    temp.x += M * Ku_shm[0][2][tx].x; temp.y += M * Ku_shm[0][2][tx].y;
                    M = ((2*mu*nu/(1-nu))*((ky*kx*(i==j)+k_i*k_j*(1==0))/kSq - (i==j)*(1==0)) - mu*((i==1)*(j==0)+(i==0)*(j==1)) - (2*mu/(1-nu))*k_i*k_j*ky*kx/kSqSq + mu*(k_i*kx*(j==1)+k_i*ky*(j==0)+k_j*kx*(i==1)+k_j*ky*(i==0))/kSq);
                    temp.x += M * Ku_shm[1][0][tx].x; temp.y += M * Ku_shm[1][0][tx].y;
                    M = ((2*mu*nu/(1-nu))*((ky*ky*(i==j)+k_i*k_j*(1==1))/kSq - (i==j)*(1==1)) - mu*((i==1)*(j==1)+(i==1)*(j==1)) - (2*mu/(1-nu))*k_i*k_j*ky*ky/kSqSq + mu*(k_i*ky*(j==1)+k_i*ky*(j==1)+k_j*ky*(i==1)+k_j*ky*(i==1))/kSq);
                    temp.x += M * Ku_shm[1][1][tx].x; temp.y += M * Ku_shm[1][1][tx].y;
                    M = ((2*mu*nu/(1-nu))*((ky*kz*(i==j)+k_i*k_j*(1==2))/kSq - (i==j)*(1==2)) - mu*((i==1)*(j==2)+(i==2)*(j==1)) - (2*mu/(1-nu))*k_i*k_j*ky*kz/kSqSq + mu*(k_i*kz*(j==1)+k_i*ky*(j==2)+k_j*kz*(i==1)+k_j*ky*(i==2))/kSq);
                    temp.x += M * Ku_shm[1][2][tx].x; temp.y += M * Ku_shm[1][2][tx].y;
                    M = ((2*mu*nu/(1-nu))*((kz*kx*(i==j)+k_i*k_j*(2==0))/kSq - (i==j)*(2==0)) - mu*((i==2)*(j==0)+(i==0)*(j==2)) - (2*mu/(1-nu))*k_i*k_j*kz*kx/kSqSq + mu*(k_i*kx*(j==2)+k_i*kz*(j==0)+k_j*kx*(i==2)+k_j*kz*(i==0))/kSq);
                    temp.x += M * Ku_shm[2][0][tx].x; temp.y += M * Ku_shm[2][0][tx].y;
                    M = ((2*mu*nu/(1-nu))*((kz*ky*(i==j)+k_i*k_j*(2==1))/kSq - (i==j)*(2==1)) - mu*((i==2)*(j==1)+(i==1)*(j==2)) - (2*mu/(1-nu))*k_i*k_j*kz*ky/kSqSq + mu*(k_i*ky*(j==2)+k_i*kz*(j==1)+k_j*ky*(i==2)+k_j*kz*(i==1))/kSq);
                    temp.x += M * Ku_shm[2][1][tx].x; temp.y += M * Ku_shm[2][1][tx].y;
                    M = ((2*mu*nu/(1-nu))*((kz*kz*(i==j)+k_i*k_j*(2==2))/kSq - (i==j)*(2==2)) - mu*((i==2)*(j==2)+(i==2)*(j==2)) - (2*mu/(1-nu))*k_i*k_j*kz*kz/kSqSq + mu*(k_i*kz*(j==2)+k_i*kz*(j==2)+k_j*kz*(i==2)+k_j*kz*(i==2))/kSq);
                    temp.x += M * Ku_shm[2][2][tx].x; temp.y += M * Ku_shm[2][2][tx].y;
#endif
            Ksig_shm[i][j][tx].x += temp.x;
            Ksig_shm[i][j][tx].y += temp.y;
        }
#ifdef DIMENSION3
        Ksig_shm[i][j][tx].x /= N*N*N;
        Ksig_shm[i][j][tx].y /= N*N*N;
#else
        Ksig_shm[i][j][tx].x /= N*N;
        Ksig_shm[i][j][tx].y /= N*N;
#endif
        *(Ku+in_idx+Lsize(L)*(i*3+j)) = Ksig_shm[i][j][tx];
        // Division for Normalization
    }
    __syncthreads();
}

#ifdef PYTHON_COMPATIBILITY_FFTW
fftw_plan plan_r2c;
fftw_plan plan_c2r;
#endif

void
setupSystem() {
#ifndef DIMENSION3
    cufftPlan2d(&g_planr2c, N, N, FORWARD_FFT);
    cufftPlan2d(&g_planc2r, N, N, BACKWARD_FFT);
#else
    cufftPlan3d(&g_planr2c, N, N, N, FORWARD_FFT);
    cufftPlan3d(&g_planc2r, N, N, N, BACKWARD_FFT);
#endif
    cublasInit();
#ifdef PYTHON_COMPATIBILITY_FFTW
    double *in = (double *)fftw_malloc(N*N*sizeof(double)); 
    fftw_complex *out = (fftw_complex *)fftw_malloc(N*N*sizeof(fftw_complex)); 
    plan_r2c = fftw_plan_dft_r2c_2d(N, N, in, out, FFTW_ESTIMATE | FFTW_UNALIGNED);
    plan_c2r = fftw_plan_dft_c2r_2d(N, N, out, in, FFTW_ESTIMATE | FFTW_UNALIGNED);
    fftw_free(in);
    fftw_free(out);
#endif
}

__host__ void
calculateSigma( data_type* u, data_type* sigma, d_dim_vector L )
{
    dim3 grid(KGridSize(L));
    dim3 tids(TILEX, 3, 3);
    cdata_type *Ku;
    CUDA_SAFE_CALL(cudaMalloc((void**) &Ku, sizeof(cdata_type)*LKsize(L)*NUM_COMP));

    // Fourier transform u
#ifndef PYTHON_COMPATIBILITY_TRANSPOSE_FFT
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            fft_r2c(g_planr2c, (fft_dtype_r*)(u+Lsize(L)*(3*i+j)), (fft_dtype_c*)(Ku+LKsize(L)*(3*i+j)));
#else
#ifdef PYTHON_COMPATIBILITY_FFTW
    double *in = (double *)fftw_malloc(N*N*sizeof(double));
    fftw_complex *out = (fftw_complex *)fftw_malloc(N*N*sizeof(fftw_complex));
#endif
    data_type *in_transpose;
    cudaMalloc((void**)&in_transpose, sizeof(data_type)*Lsize(L));
    cdata_type *out_transpose;
    cudaMalloc((void**)&out_transpose, sizeof(cdata_type)*LKsize(L));
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++) {
            dim3 grid((L.x+BLOCK_DIM-1)/BLOCK_DIM, (L.y+BLOCK_DIM-1)/BLOCK_DIM);
            dim3 tids(BLOCK_DIM, BLOCK_DIM);
            dim3 kgrid(((N/2+1)+BLOCK_DIM-1)/BLOCK_DIM, (N+BLOCK_DIM-1)/BLOCK_DIM);
            dim3 ktids(BLOCK_DIM, BLOCK_DIM);
            transpose<data_type><<<grid, tids>>>(in_transpose, u+Lsize(L)*(3*i+j), L.x, L.y);
#ifdef PYTHON_COMPATIBILITY_FFTW
            cudaMemcpy(in, in_transpose, Lsize(L)*sizeof(double), cudaMemcpyDeviceToHost);
            fftw_execute_dft_r2c(plan_r2c, in, out);
            cudaMemcpy(out_transpose, out, (L.y/2+1)*L.x*sizeof(fftw_complex), cudaMemcpyHostToDevice);
#else
            fft_r2c(g_planr2c, (fft_dtype_r*)(in_transpose), (fft_dtype_c*)(out_transpose));
#endif
            transpose<cdata_type><<<kgrid, ktids>>>(Ku+LKsize(L)*(3*i+j), out_transpose, L.y/2+1, L.x);
        }
    cudaFree(in_transpose);
    cudaFree(out_transpose);
    cudaThreadSynchronize();
#endif
    
    // calculateKSigma
#ifndef PYTHON_COMPATIBILITY_TRANSPOSE_FFT
    d_dim_vector newL = L;
    newL.x = L.x/2+1;
    calculateKSigma<<<grid, tids>>>(Ku, newL);
#else
    dim3 ngrid((N+TILEX-1)/TILEX, N/2+1);
    dim3 ntids(TILEX, 3, 3);
    d_dim_vector newL = L;
    newL.y = L.y/2+1;
    calculateKSigma<<<ngrid, ntids>>>(Ku, newL);//, L.y/2+1);
#endif

    // inverse Fourier kSigma
#ifndef PYTHON_COMPATIBILITY_TRANSPOSE_FFT
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            fft_c2r(g_planc2r, (fft_dtype_c*)(Ku+LKsize(L)*(3*i+j)), (fft_dtype_r*)(sigma+Lsize(L)*(3*i+j)));
#else
    cudaThreadSynchronize();
    cudaMalloc((void**)&in_transpose, sizeof(data_type)*Lsize(L));
    cudaMalloc((void**)&out_transpose, sizeof(cdata_type)*LKsize(L));
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++) {
            dim3 grid((L.x+BLOCK_DIM-1)/BLOCK_DIM, (L.y+BLOCK_DIM-1)/BLOCK_DIM);
            dim3 tids(BLOCK_DIM, BLOCK_DIM);
            dim3 kgrid((N+BLOCK_DIM-1)/BLOCK_DIM, ((N/2+1)+BLOCK_DIM-1)/BLOCK_DIM);
            dim3 ktids(BLOCK_DIM, BLOCK_DIM);
            transpose<cdata_type><<<kgrid,ktids>>>(out_transpose, Ku+(L.y/2+1)*L.x*(3*i+j), L.x, L.y/2+1);
#ifdef PYTHON_COMPATIBILITY_FFTW
            cudaMemcpy(out, out_transpose, (L.x/2+1)*L.y*sizeof(fftw_complex), cudaMemcpyDeviceToHost);
            fftw_execute_dft_c2r(plan_c2r, out, in);
            cudaMemcpy(in_transpose, in, Lsize(L)*sizeof(double), cudaMemcpyHostToDevice);
#else
            fft_c2r(g_planc2r, (fft_dtype_c*)(out_transpose), (fft_dtype_r*)(in_transpose));
#endif
            transpose<data_type><<<grid,tids>>>(sigma+Lsize(L)*(3*i+j),in_transpose, L.x, L.y);
        }
#ifdef PYTHON_COMPATIBILITY_FFTW
    fftw_free(in);
    fftw_free(out);
#endif
    cudaFree(in_transpose);
    cudaFree(out_transpose);
#endif
    CUDA_SAFE_CALL(cudaFree(Ku));
    //printf("max sigma %lf\n", reduceMax(sigma, Lsize(L)*9));
}

#ifdef LOADING
__global__ void
loadSigma( data_type t, data_type* sigma, d_dim_vector L )
{
    int bx = blockIdx.x;     int by = blockIdx.y;
    /* x coordinate is split into threads */
    int tx = threadIdx.x;    
    /* Indices of the array this thread will tackle */
    int i = threadIdx.y;
    int j = threadIdx.z;
    //int idx = i*3+j;
    int ix = bx*TILEX + tx;
    int in_idx = by*L.x + ix;
     
    const data_type load[3][3] = LOAD_DEF;

    // matt
    *(sigma+in_idx+(i*3+j)*Lsize(L)) += 2.*mu*load[i][j]*LOADING_RATE*t; 
    if (i==j)
        *(sigma+in_idx+(i*3+j)*Lsize(L)) += 2.*mu*nu/(1-2*nu)*(load[0][0]+load[1][1]+load[2][2])*LOADING_RATE*t; 
        //*(sigma+in_idx+(i*3+j)*Lsize(L)) += (load[0][0]+load[1][1]+load[2][2])*LOADING_RATE; 
}
#endif

__host__ void
calculateFlux( data_type t, data_type* u, data_type* rhs, data_type* velocity, d_dim_vector L )
{
    dim3 grid(GridSize(L));
    dim3 tids(TILEX, NUM_COMP);
    dim3 tidt(TILEX, 3,3);
    data_type *sigma;
    CUDA_SAFE_CALL(cudaMalloc((void**) &sigma, sizeof(data_type)*Lsize(L)*NUM_SIG_COMP));

    calculateSigma(u, sigma, L);
    cudaThreadSynchronize();

#ifdef LOADING
    loadSigma<<<grid, tidt>>>(t, sigma, L);
    cudaThreadSynchronize();
#endif

    // calculate flux
    centralHJ<<<grid, tids>>>(u, sigma, rhs, velocity, L);
    cudaThreadSynchronize();

#ifdef DYNAMIC_NUCLEATION
    updateField(rhs, 1.0/20, beta0dot, Lsize(L));
#endif

    CUDA_SAFE_CALL(cudaFree(sigma));
}

__host__ data_type
reduceMax( data_type* u, int size )
{
    int idx = Iamax(size, u, 1);
    data_type maxVel = 0.;
    CUDA_SAFE_CALL(cudaMemcpy(&maxVel, u+idx-1,  sizeof(data_type), cudaMemcpyDeviceToHost));
    return maxVel;
}

__host__ void
updateField( data_type* u, data_type timeStep, data_type *rhs, int size)
{
    axpy(NUM_COMP*size, timeStep, rhs, 1, u, 1);
}

__host__ double
simpleTVD( data_type* u, d_dim_vector L, data_type time, data_type endTime)
{
    data_type *rhs, *velocity;
    double timestep = 0.;
    CUDA_SAFE_CALL(cudaMalloc((void**) &rhs, sizeof(data_type)*Lsize(L)*NUM_COMP));
    CUDA_SAFE_CALL(cudaMalloc((void**) &velocity, sizeof(data_type)*Lsize(L)));

    calculateFlux(time, u, rhs, velocity, L);
    timestep = CFLsafeFactor / reduceMax(velocity, Lsize(L)) / N;

#ifdef DYNAMIC_NUCLEATION
    if (timestep > maxNucleationTimestep)
        timestep = maxNucleationTimestep;
#endif

    if (time+timestep > endTime)
        timestep = endTime - time + ME;

    updateField(u, timestep, rhs, Lsize(L));
    CUDA_SAFE_CALL(cudaFree(rhs));
    CUDA_SAFE_CALL(cudaFree(velocity));
#ifdef VACANCIES
    calculateDiffusion(u, L, timestep);
#endif
    return timestep;
} 

__host__ double
TVD3rd( data_type* u, d_dim_vector L, data_type time, data_type endTime)
{
    data_type *rhs, *velocity;
    data_type *L0, *L1, *L2;
    data_type *F0, *F1, *F2;
    double timestep = 0.;
    const data_type alpha[3][3] = {{1.,0.,0.}, {3./4.,1./4.,0.}, {1./3.,0.,2./3.}};
    const data_type beta[3][3] = {{1.,0.,0.}, {0.,1./4.,0.}, {0.,0.,2./3.}};

    CUDA_SAFE_CALL(cudaMalloc((void**) &F0, sizeof(data_type)*Lsize(L)*NUM_COMP));
    CUDA_SAFE_CALL(cudaMalloc((void**) &F1, sizeof(data_type)*Lsize(L)*NUM_COMP));
    CUDA_SAFE_CALL(cudaMalloc((void**) &F2, sizeof(data_type)*Lsize(L)*NUM_COMP));
    cudaMemset(F0, 0, sizeof(data_type)*Lsize(L)*NUM_COMP);
    cudaMemset(F1, 0, sizeof(data_type)*Lsize(L)*NUM_COMP);
    cudaMemset(F2, 0, sizeof(data_type)*Lsize(L)*NUM_COMP);
    CUDA_SAFE_CALL(cudaMalloc((void**) &L0, sizeof(data_type)*Lsize(L)*NUM_COMP));
    CUDA_SAFE_CALL(cudaMalloc((void**) &L1, sizeof(data_type)*Lsize(L)*NUM_COMP));
    CUDA_SAFE_CALL(cudaMalloc((void**) &L2, sizeof(data_type)*Lsize(L)*NUM_COMP));
    cudaMemset(L0, 0, sizeof(data_type)*Lsize(L)*NUM_COMP);
    cudaMemset(L1, 0, sizeof(data_type)*Lsize(L)*NUM_COMP);
    cudaMemset(L2, 0, sizeof(data_type)*Lsize(L)*NUM_COMP);
    CUDA_SAFE_CALL(cudaMalloc((void**) &rhs, sizeof(data_type)*Lsize(L)*NUM_COMP));
    CUDA_SAFE_CALL(cudaMalloc((void**) &velocity, sizeof(data_type)*Lsize(L)));

    calculateFlux(time, u, rhs, velocity, L);
    timestep = CFLsafeFactor / reduceMax(velocity, Lsize(L)) / N;
    if (time+timestep > endTime)
        timestep = endTime - time + ME;

    updateField(L0, timestep, rhs, Lsize(L));
    updateField(F0, alpha[0][0], u, Lsize(L));
    updateField(F0, beta[0][0], L0, Lsize(L));

    calculateFlux(time, F0, rhs, velocity, L);

    updateField(L1, timestep, rhs, Lsize(L));
    updateField(F1, alpha[1][0], u, Lsize(L));
    updateField(F1, beta[1][0], L0, Lsize(L));
    updateField(F1, alpha[1][1], F0, Lsize(L));
    updateField(F1, beta[1][1], L1, Lsize(L));

    calculateFlux(time, F1, rhs, velocity, L);

    updateField(L2, timestep, rhs, Lsize(L));
    updateField(u, alpha[2][0]-1, u, Lsize(L));
    updateField(u, beta[2][0], L0, Lsize(L));
    updateField(u, alpha[2][1], F0, Lsize(L));
    updateField(u, beta[2][1], L1, Lsize(L));
    updateField(u, alpha[2][2], F1, Lsize(L));
    updateField(u, beta[2][2], L2, Lsize(L));

    //vec_copy(Lsize(L)*9, F2, 1, u, 1);
    CUDA_SAFE_CALL(cudaFree(F0));
    CUDA_SAFE_CALL(cudaFree(F1));
    CUDA_SAFE_CALL(cudaFree(F2));
    CUDA_SAFE_CALL(cudaFree(L0));
    CUDA_SAFE_CALL(cudaFree(L1));
    CUDA_SAFE_CALL(cudaFree(L2));
    CUDA_SAFE_CALL(cudaFree(rhs));
    CUDA_SAFE_CALL(cudaFree(velocity));
#ifdef VACANCIES
    // NOTE: python uses one flux calculation for cfield whereas we
    // instead use 3 here.  The diffusion does follow the original prescription.
    calculateDiffusion(u, L, timestep);
#endif
    return timestep;
} 

__host__ void
runTVDSimple( data_type* u, d_dim_vector L, double time, double endTime )
{
    while(time < endTime) {
        double timeStep = simpleTVD(u, L, time, endTime);
        printf("%f +%f\n", time, timeStep);
        time += timeStep;
    }
}

__host__ void
runTVD( data_type* u, d_dim_vector L, double time, double endTime )
{
    while(time < endTime) {
        double timeStep = TVD3rd(u, L, time, endTime);
        printf("%f +%f\n", time, timeStep);
        time += timeStep;
    }
}

#endif // #ifndef _PLASTICITY_KERNEL_H_
