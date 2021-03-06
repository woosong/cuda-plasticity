/* plasticity.h
 */

#ifndef _PLASTICITY_H_
#define _PLASTICITY_H_

#define NUM_COMP 9
#define NUM_SIG_COMP 9

//#define TVDstep TVD3rd
#define TVDstep simpleTVD
#define LAST_SHOT_ONLY

//#define DEBUG_TIMESTEPS
//#define PYTHON_COMPATIBILITY
//#define PYTHON_COMPATIBILITY_TRANSPOSE_FFT
//#define PYTHON_COMPATIBILITY_FFTW
// Python compatibility divide does give different results...
#define PYTHON_COMPATIBILITY_DIVIDE
//Not relevant for python compatibility divide mode
//#define SLOPPY_NO_DIVIDE_BY_ZERO
#define CONTINUE_RUN

//#define NEWGLIDEONLY
//#define RUN_DESC "ngo"

//#define SLIPSYSTEMS
//#define mixing 0.9

#ifdef DEBUG_TIMESTEPS
#define RUN_DESC "debug"
#endif

// Configuration
// By default, 2D. 
//#define DIMENSION3

#ifdef DIMENSION3
#define FILE_PREFIX "3d_"
#else
#define FILE_PREFIX ""
#endif

#define N 128 
#define lambda 1
#define CFLsafeFactor 0.5
#define mu 0.5
#define nu 0.3
//#define LENGTHSCALE

//#define DYNAMIC_NUCLEATION

//#define VACANCIES
#ifdef VACANCIES
#define PI2              2*3.14159265
#define NUM_COMP         10
#define VACANCY_COMP     9
#define vacancycost      1e2  
#define vacancydiffusion 1e-4
#define FILE_PREFIX2 "vac_"
#else
#define FILE_PREFIX2 ""
#endif

#define theta 1.0

#define DOUBLE
#define LLF

#define LOADING
#define UNIAXIAL_ZZ
//#define UNIAXIAL_XX
//#define UNIAXIAL_YY
//#define COLDROLLING_XY
//#define COLDROLLING_YZ
//strain loading rate
#define LOADING_RATE 0.0005
//#define LOADING_RATE 0.001

#ifdef UNIAXIAL_ZZ
// Uniaxial ZZ Loading
#define LOAD_DEF {{-0.5,0,0},{0,-0.5,0},{0,0,1}} 
#define RUN_DESC "unizz"
#endif
#ifdef UNIAXIAL_XX
// Uniaxial XX
#define LOAD_DEF {{1.0,0,0},{0,-0.5,0},{0,0,-0.5}}
#define RUN_DESC "unixx"
#endif
#ifdef UNIAXIAL_YY
// Uniaxial YY
#define LOAD_DEF {{-0.5,0,0},{0,1.0,0},{0,0,-0.5}}
#define RUN_DESC "uniyy"
#endif
#ifdef COLDROLLING_XY
// Cold Rolling XY
#define LOAD_DEF {{1.0,0,0},{0,-1.0,0},{0,0,0.0}}
#define RUN_DESC "coldxy"
#endif
#ifdef COLDROLLING_YZ
// Cold Rolling YZ
#define LOAD_DEF {{0.0,0,0},{0,1.0,0},{0,0,-1.0}}
#define RUN_DESC "coldyz"
#endif

#define xstr(s) str(s)
#define str(s) #s 
#ifdef SLIPSYSTEMS
#define MIX_DESC ""
#ifdef mixing
#undef MIX_DESC
#define MIX_DESC "_mix" xstr(mixing)
#endif
#define THETA_DESC ""
#ifdef theta
#undef THETA_DESC
#define THETA_DESC "_t" xstr(theta)
#endif
#define RELAX_RUN_DESC "sliprelax" MIX_DESC THETA_DESC
#else
#define RELAX_RUN_DESC "relax"
#endif
#ifndef RUN_DESC
#define RUN_DESC RELAX_RUN_DESC
#endif

// From here on defines data type related definitions
#ifndef DOUBLE
#define ReadMatrixFileFunc ReadMatrixFile
#define PRECISION_STR "sp"
#define data_type float
#define cdata_type cuComplex
#define init_cdata(x,y) make_cuFloatComplex(x,y)
#define FORWARD_FFT CUFFT_R2C
#define BACKWARD_FFT CUFFT_C2R
#define fft_r2c cufftExecR2C
#define fft_c2r cufftExecC2R
#define fft_dtype_r cufftReal
#define fft_dtype_c cufftComplex
#define Iamax cublasIsamax
#define axpy cublasSaxpy
#define vec_copy cublasScopy
#define ME 1.0e-8
//#define ME 1.1920929e-7
#else
#define ReadMatrixFileFunc ReadDoubleMatrixFile
#define PRECISION_STR "dp"
#define data_type double
#define cdata_type cuDoubleComplex
#define init_cdata(x,y) make_cuDoubleComplex(x,y)
#define FORWARD_FFT CUFFT_D2Z
#define BACKWARD_FFT CUFFT_Z2D
#define fft_r2c cufftExecD2Z
#define fft_c2r cufftExecZ2D
#define fft_dtype_r cufftDoubleReal
#define fft_dtype_c cufftDoubleComplex
#define Iamax cublasIdamax
#define axpy cublasDaxpy
#define vec_copy cublasDcopy
#define ME 2.2204460492503131e-16
#endif

#ifndef DIMENSION3
typedef int2 d_dim_vector;
#define Lsize(L) (L.x*L.y)
#define LKsize(L) ((L.x/2+1)*L.y)
#define GridSize(L) ((L.x-1)/TILEX+1), L.y 
#define KGridSize(L) ((L.x/2)/TILEX)+1, L.y
#define locate(f,v,idx) (*(f+((idx)*L.y+((v.y+L.y)%L.y))*L.x+((v.x+L.x)%L.x)))
#define locateop(f,v,op,d,idx) (*(f+((idx)*L.y+((v.y op d.y +L.y)%L.y))*L.x+((v.x op d.x +L.x)%L.x)))
#else
typedef int3 d_dim_vector;
#define Lsize(L) (L.x*L.y*L.z)
#define LKsize(L) ((L.x/2+1)*L.y*L.z)
#define GridSize(L) ((L.x-1)/TILEX+1), L.y*L.z
#define KGridSize(L) ((L.x/2)/TILEX)+1, L.y*L.z
#define locate(f,v,idx) (*(f+(((idx)*L.z+((v.z+L.z)%L.z))*L.y+((v.y+L.y)%L.y))*L.x+((v.x+L.x)%L.x)))
#define locateop(f,v,op,d,idx) (*(f+(((idx)*L.z+(v.z op d.z +L.z)%L.z)*L.y+((v.y op d.y +L.y)%L.y))*L.x+((v.x op d.x +L.x)%L.x)))
#endif

#ifdef DYNAMIC_NUCLEATION
extern data_type maxNucleationTimestep;
extern data_type *beta0dot;
#endif

#endif // _PLASTICITY_H_

