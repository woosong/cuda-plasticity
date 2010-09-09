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
findDerivatives( data_type* u, int i, int j, d_dim_vector x, int coord, data_type *deriv_p, data_type *deriv_m, d_dim_vector L)
{
    volatile d_dim_vector d;
    d.x = (coord==0);
    d.y = (coord==1);
#ifdef DIMENSION3
    d.z = (coord==2);
#endif
    volatile int idx = i*3+j;
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
centralHJ( data_type* u, data_type* sig, data_type* rhs, data_type* velocity, d_dim_vector L )
{
    volatile int bx = blockIdx.x;     
    int by = blockIdx.y;
#ifdef DIMENSION3
    int bz = blockIdx.z;
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
    __shared__ data_type du[NUM_ELEM][3][3][TILEX];

    // these have only one among nine
    // we only need 6 for 3D, but make 8 for using for rhomod
    __shared__ data_type a[NUM_ELEM2][TILEX];
    
    // Specific to plasticity
    // Use a for rhomod, since it's only used in preparation
    //__shared__ data_type rhomod[4][TILEX];
    __shared__ data_type sigma[3][3][TILEX];
    __shared__ data_type v[NUM_ELEM2][3][TILEX];

    d_dim_vector x;
    x.x = ix;
    x.y = by;
#ifdef DIMENSION3
    x.z = bz;
#endif
    // Determine the derivatives
    findDerivatives(u, i, j, x, 0, &du[0][i][j][tx], &du[1][i][j][tx], L);
    findDerivatives(u, i, j, x, 1, &du[2][i][j][tx], &du[3][i][j][tx], L);
#ifdef DIMENSION3
    findDerivatives(u, i, j, x, 2, &du[4][i][j][tx], &du[5][i][j][tx], L);
#endif
    sigma[i][j][tx] = locate(sig, x, idx);
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
        // rhomod += ux[i,j]*ux[i,j] when i!=x
        rhomod += du[kx][1][0][tx]*du[kx][1][0][tx];
        rhomod += du[kx][1][1][tx]*du[kx][1][1][tx];
        rhomod += du[kx][1][2][tx]*du[kx][1][2][tx];
        rhomod += du[kx][2][0][tx]*du[kx][2][0][tx];
        rhomod += du[kx][2][1][tx]*du[kx][2][1][tx];
        rhomod += du[kx][2][2][tx]*du[kx][2][2][tx];

        // rhomod += uy[i,j]*uy[i,j] when i!=y
        rhomod += du[ky][0][0][tx]*du[ky][0][0][tx];
        rhomod += du[ky][0][1][tx]*du[ky][0][1][tx];
        rhomod += du[ky][0][2][tx]*du[ky][0][2][tx];
        rhomod += du[ky][2][0][tx]*du[ky][2][0][tx];
        rhomod += du[ky][2][1][tx]*du[ky][2][1][tx];
        rhomod += du[ky][2][2][tx]*du[ky][2][2][tx];

#ifdef DIMENSION3
        // rhomod += uz[z,j]*uz[z,j] when i!=z
        rhomod += du[kz][0][0][tx]*du[kz][0][0][tx];
        rhomod += du[kz][0][1][tx]*du[kz][0][1][tx];
        rhomod += du[kz][0][2][tx]*du[kz][0][2][tx];
        rhomod += du[kz][1][0][tx]*du[kz][1][0][tx];
        rhomod += du[kz][1][1][tx]*du[kz][1][1][tx];
        rhomod += du[kz][1][2][tx]*du[kz][1][2][tx];
#endif

        // rhomod -= 2*ux[y,j]*uy[x,j]
        rhomod -= 2*du[kx][1][0][tx]*du[ky][0][0][tx];
        rhomod -= 2*du[kx][1][1][tx]*du[ky][0][1][tx];
        rhomod -= 2*du[kx][1][2][tx]*du[ky][0][2][tx];
#ifdef DIMENSION3
        // rhomod -= 2*ux[z,j]*uz[x,j]
        rhomod -= 2*du[kx][2][0][tx]*du[kz][0][0][tx];
        rhomod -= 2*du[kx][2][1][tx]*du[kz][0][1][tx];
        rhomod -= 2*du[kx][2][2][tx]*du[kz][0][2][tx];
        // rhomod -= 2*uy[z,j]*uz[y,j]
        rhomod -= 2*du[ky][2][0][tx]*du[kz][1][0][tx];
        rhomod -= 2*du[ky][2][1][tx]*du[kz][1][1][tx];
        rhomod -= 2*du[ky][2][2][tx]*du[kz][1][2][tx];
#endif

        if (rhomod < 0.)
            rhomod = 0.;

        // store in shm
        a[idx][tx] = sqrt(rhomod);
#ifndef DIMENSION3
    } else {
        if (idx<8) {
            volatile int tidx = idx-4;
#else
    } {
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
            vx = du[ky][0][0][tx]*sigma[1][0][tx];
            vx -= du[kx][1][0][tx]*sigma[1][0][tx];
            vx -= du[kx][2][0][tx]*sigma[2][0][tx];
            vx += du[ky][0][1][tx]*sigma[1][1][tx];
            vx -= du[kx][1][1][tx]*sigma[1][1][tx];
            vx -= du[kx][2][1][tx]*sigma[2][1][tx];
            vx += du[ky][0][2][tx]*sigma[1][2][tx];
            vx -= du[kx][1][2][tx]*sigma[1][2][tx];
            vx -= du[kx][2][2][tx]*sigma[2][2][tx];
#ifdef DIMENSION3
            // v[x] += uz[x][n]*sigma[z][n]
            vx += du[kz][0][0][tx]*sigma[2][0][tx];
            vx += du[kz][0][1][tx]*sigma[2][1][tx];
            vx += du[kz][0][2][tx]*sigma[2][2][tx];
#endif
            // v[y] += ux[y][n]*sigma[x][n]
            // v[y] -= uy[x][n]*sigma[x][n]
            // v[y] -= uy[z][n]*sigma[z][n]
            vy = du[kx][1][0][tx]*sigma[0][0][tx];
            vy -= du[ky][0][0][tx]*sigma[0][0][tx];
            vy -= du[ky][2][0][tx]*sigma[2][0][tx];
            vy += du[kx][1][1][tx]*sigma[0][1][tx];
            vy -= du[ky][0][1][tx]*sigma[0][1][tx];
            vy -= du[ky][2][1][tx]*sigma[2][1][tx];
            vy += du[kx][1][2][tx]*sigma[0][2][tx];
            vy -= du[ky][0][2][tx]*sigma[0][2][tx];
            vy -= du[ky][2][2][tx]*sigma[2][2][tx];
#ifdef DIMENSION3
            // v[y] += uz[y][n]*sigma[z][n]
            vy += du[kz][1][0][tx]*sigma[2][0][tx];
            vy += du[kz][1][1][tx]*sigma[2][1][tx];
            vy += du[kz][1][2][tx]*sigma[2][2][tx];
#endif

            // v[z] += ux[z][n]*sigma[x][n]
            // v[z] += uy[z][n]*sigma[y][n]
            vz = du[kx][2][0][tx]*sigma[0][0][tx];
            vz += du[ky][2][0][tx]*sigma[1][0][tx];
            vz += du[kx][2][1][tx]*sigma[0][1][tx];
            vz += du[ky][2][1][tx]*sigma[1][1][tx];
            vz += du[kx][2][2][tx]*sigma[0][2][tx];
            vz += du[ky][2][2][tx]*sigma[1][2][tx];
#ifdef DIMENSION3
            // v[z] -= uz[x][n]*sigma[x][n]
            // v[z] -= uz[y][n]*sigma[y][n]
            // these two cancel
            // v[z] += uz[z][n]*sigma[z][n]
            // v[z] -= uz[z][n]*sigma[z][n]
            vz -= du[kz][0][0][tx]*sigma[0][0][tx];
            vz -= du[kz][0][1][tx]*sigma[0][1][tx];
            vz -= du[kz][0][2][tx]*sigma[0][2][tx];
            vz -= du[kz][1][0][tx]*sigma[2][0][tx];
            vz -= du[kz][1][1][tx]*sigma[2][1][tx];
            vz -= du[kz][1][2][tx]*sigma[2][2][tx];
#endif

            // FIXME - glide term
            data_type sigma_tr = (sigma[0][0][tx]+sigma[1][1][tx]+sigma[2][2][tx])/3.*lambda;
            vx += sigma_tr*(du[kx][0][0][tx]+du[kx][1][1][tx]+du[kx][2][2][tx]);
            vy += sigma_tr*(du[ky][0][0][tx]+du[ky][1][1][tx]+du[ky][2][2][tx]);
            vx -= sigma_tr*(du[kx][0][0][tx]+du[ky][0][1][tx]);
            vy -= sigma_tr*(du[kx][1][0][tx]+du[ky][1][1][tx]);
            vz -= sigma_tr*(du[kx][2][0][tx]+du[ky][2][1][tx]);
#ifdef DIMENSION3
            vz += sigma_tr*(du[kz][0][0][tx]+du[kz][1][1][tx]+du[kz][2][2][tx]);
            vx -= sigma_tr*(du[kz][0][2][tx]);
            vy -= sigma_tr*(du[kz][1][2][tx]);
            vz -= sigma_tr*(du[kz][2][2][tx]);
#endif
            // store in shm
            v[tidx][0][tx] = vx;
            v[tidx][1][tx] = vy;
            v[tidx][2][tx] = vz;
        }
    }
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
        for(int k = 0; k<NUM_ELEM2; k++) {
            volatile data_type t = v[k][idx][tx];
            if (max < t) 
                max = t;
            if (max < -t)
                max = -t;
        }
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
    for(int k=0; k<NUM_ELEM2; k++) {
        volatile data_type h;
        volatile int kx = k%2;
#ifndef DIMENSION3
        volatile int ky = k/2+2;
#else
        volatile int ky = (k%4)/2+2;
        volatile int kz = k/4+4;
#endif
        // H_ij = v_l (d_l beta_ij - d_i beta_lj)
        h = du[kx][i][j][tx] * v[k][0][tx] + du[ky][i][j][tx] * v[k][1][tx];
#ifdef DIMENSION3
        h += du[kz][i][j][tx] * v[k][2][tx];
#endif
        if (i==0) {
            h -= v[k][0][tx] * du[kx][0][j][tx];
            h -= v[k][1][tx] * du[kx][1][j][tx];
            h -= v[k][2][tx] * du[kx][2][j][tx];
        } else {
            if (i==1) {
                h -= v[k][0][tx] * du[ky][0][j][tx];
                h -= v[k][1][tx] * du[ky][1][j][tx];
                h -= v[k][2][tx] * du[ky][2][j][tx];
            }
#ifdef DIMENSION3
            else {
                if (i==2) {
                    h -= v[k][0][tx] * du[kz][0][j][tx];
                    h -= v[k][1][tx] * du[kz][1][j][tx];
                    h -= v[k][2][tx] * du[kz][2][j][tx];
                }
            }
#endif
        }
        // Checked up until this point
        // Glide only correction
        // H_ij += lambda v_l / 3 (d_l beta_kk - d_k beta_lk)
        if (i==j) {
            h -= (v[k][0][tx]*(du[kx][1][1][tx]+du[kx][2][2][tx]-du[ky][0][1][tx])
                    +v[k][1][tx]*(du[ky][0][0][tx]+du[ky][2][2][tx]-du[kx][1][0][tx])
                    -v[k][2][tx]*(du[kx][2][0][tx]+du[ky][2][1][tx])
#ifdef DIMENSION3
                    -v[k][0][tx]*(du[kz][0][2][tx])
                    -v[k][1][tx]*(du[kz][1][2][tx])
                    +v[k][2][tx]*(du[kz][0][0][tx]+du[kz][1][1][tx]) 
#endif
                )/3.*lambda;
        }    
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
        derivative += (du[0][i][j][tx]-du[1][i][j][tx])*a[0][tx]*a[1][tx]/ax;
    //if (i!=1)
        derivative += (du[2][i][j][tx]-du[3][i][j][tx])*a[2][tx]*a[3][tx]/ay;
#ifdef DIMENSION3
        derivative += (du[4][i][j][tx]-du[5][i][j][tx])*a[4][tx]*a[5][tx]/az;
#endif
    *(rhs+idx*Lsize(L)+in_idx) = derivative;
    if (idx==0)
        *(velocity+in_idx) = ax+ay
#ifdef DIMENSION3
                            +az
#endif
                             ;
}

__global__ void
calculateKSigma( cdata_type* Ku, d_dim_vector L )
{
    // This kernel calculates the sigma in k-space
    // Since K-space field is supposedly unnecessary afterwards,
    // it is overwritten
    int bx = blockIdx.x;     
    int by = blockIdx.y;
#ifdef DIMENSION3
    int bz = blockIdx.z;
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
#endif
    data_type kSqSq = kSq*kSq;
    data_type k[3];

#ifdef PYTHON_COMPATIBILITY
    kSq += ME;
    kSqSq += ME;
#endif

    __shared__ cdata_type Ku_shm[3][3][TILEX];
    __shared__ cdata_type Ksig_shm[3][3][TILEX];
 
    k[0] = kx;
    k[1] = ky;
#ifndef DIMENSION3
    k[2] = 0;
#else
    k[2] = kz;
#endif
    if (ix < L.x) {
        Ku_shm[i][j][tx] = *(Ku+in_idx+Lsize(L)*(i*3+j));
        Ksig_shm[i][j][tx] = init_cdata(0.,0.);
    }
    __syncthreads();

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
        if (ix < L.x) {
            for(int m = 0; m < 3; m++)
                for(int n = 0; n < 3; n++) {
                    data_type M = ((2*mu*nu/(1-nu))*((k[m]*k[n]*(i==j)+k[i]*k[j]*(m==n))/kSq - (i==j)*(m==n)) - mu*((i==m)*(j==n)+(i==n)*(j==m)) - (2*mu/(1-nu))*k[i]*k[j]*k[m]*k[n]/kSqSq + mu*(k[i]*k[n]*(j==m)+k[i]*k[m]*(j==n)+k[j]*k[n]*(i==m)+k[j]*k[m]*(i==n))/kSq);
                    Ksig_shm[i][j][tx].x += M * Ku_shm[m][n][tx].x;
                    Ksig_shm[i][j][tx].y += M * Ku_shm[m][n][tx].y;
                }
        }
    }
    if (ix < L.x) {
        Ksig_shm[i][j][tx].x /= N*N;
        Ksig_shm[i][j][tx].y /= N*N;
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
    CUDA_SAFE_CALL(cudaMalloc((void**) &Ku, sizeof(cdata_type)*LKsize(L)*9));

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
    calculateKSigma<<<ngrid, ntids>>>(Ku, L.x, L.y/2+1);
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

    *(sigma+in_idx+(i*3+j)*Lsize(L)) += 2.*mu*load[i][j]*LOADING_RATE*t; 
    if (i==j)
        *(sigma+in_idx+(i*3+j)*Lsize(L)) += 2.*mu*nu/(1-2*nu)*(load[0][0]+load[1][1]+load[2][2])*LOADING_RATE*t; 
}
#endif

__host__ void
calculateFlux( data_type t, data_type* u, data_type* rhs, data_type* velocity, d_dim_vector L )
{
    dim3 grid(GridSize(L));
    dim3 tids(TILEX, 3, 3);
    data_type *sigma;
    CUDA_SAFE_CALL(cudaMalloc((void**) &sigma, sizeof(data_type)*Lsize(L)*9));

    calculateSigma(u, sigma, L);
    cudaThreadSynchronize();

#ifdef LOADING
    loadSigma<<<grid, tids>>>(t, sigma, L);
    cudaThreadSynchronize();
#endif

    // calculate flux
    centralHJ<<<grid, tids>>>(u, sigma, rhs, velocity, L);

    cudaThreadSynchronize();

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
    axpy(9*size, timeStep, rhs, 1, u, 1);
}

__host__ double
simpleTVD( data_type* u, d_dim_vector L, data_type time, data_type endTime)
{
    data_type *rhs, *velocity;
    double timestep = 0.;
    CUDA_SAFE_CALL(cudaMalloc((void**) &rhs, sizeof(data_type)*Lsize(L)*9));
    CUDA_SAFE_CALL(cudaMalloc((void**) &velocity, sizeof(data_type)*Lsize(L)));

    calculateFlux(time, u, rhs, velocity, L);
    timestep = CFLsafeFactor / reduceMax(velocity, Lsize(L)) / N;
    if (time+timestep > endTime)
        timestep = endTime - time + ME;

    updateField(u, timestep, rhs, Lsize(L));
    CUDA_SAFE_CALL(cudaFree(rhs));
    CUDA_SAFE_CALL(cudaFree(velocity));
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

    CUDA_SAFE_CALL(cudaMalloc((void**) &F0, sizeof(data_type)*Lsize(L)*9));
    CUDA_SAFE_CALL(cudaMalloc((void**) &F1, sizeof(data_type)*Lsize(L)*9));
    CUDA_SAFE_CALL(cudaMalloc((void**) &F2, sizeof(data_type)*Lsize(L)*9));
    cudaMemset(F0, 0, sizeof(data_type)*Lsize(L)*9);
    cudaMemset(F1, 0, sizeof(data_type)*Lsize(L)*9);
    cudaMemset(F2, 0, sizeof(data_type)*Lsize(L)*9);
    CUDA_SAFE_CALL(cudaMalloc((void**) &L0, sizeof(data_type)*Lsize(L)*9));
    CUDA_SAFE_CALL(cudaMalloc((void**) &L1, sizeof(data_type)*Lsize(L)*9));
    CUDA_SAFE_CALL(cudaMalloc((void**) &L2, sizeof(data_type)*Lsize(L)*9));
    cudaMemset(L0, 0, sizeof(data_type)*Lsize(L)*9);
    cudaMemset(L1, 0, sizeof(data_type)*Lsize(L)*9);
    cudaMemset(L2, 0, sizeof(data_type)*Lsize(L)*9);
    CUDA_SAFE_CALL(cudaMalloc((void**) &rhs, sizeof(data_type)*Lsize(L)*9));
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
