/* plasticity.h
 */

#ifndef _PLASTICITY_H_
#define _PLASTICITY_H_

//#define DEBUG_TIMESTEPS
#define PYTHON_COMPATIBILITY
#define PYTHON_COMPATIBILITY_TRANSPOSE_FFT
//#define PYTHON_COMPATIBILITY_FFTW
// Python compatibility divide does give different results...
#define PYTHON_COMPATIBILITY_DIVIDE
//Not relevant for python compatibility divide mode
//#define SLOPPY_NO_DIVIDE_BY_ZERO

#ifdef DEBUG_TIMESTEPS
#define RUN_DESC "debug"
#endif

// Configuration
#define N 512
#define lambda 1
#define CFLsafeFactor 0.5
#define mu 0.5
#define nu 0.3

#define theta 1.0

#define DOUBLE
//#define LLF

#define LOADING
#define UNIAXIAL_ZZ
//#define UNIAXIAL_XX
//#define COLDROLLING_XY
//#define COLDROLLING_YZ
//strain loading rate
#define LOADING_RATE 0.05

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
#define RUN_DESC "unixy"
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

#define RELAX_RUN_DESC "relax"
#ifndef RUN_DESC
#define RUN_DESC RELAX_RUN_DESC
#endif

// From here on defines data type related definitions
#ifndef DOUBLE
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

#endif // _PLASTICITY_H_

