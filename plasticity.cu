/* plasticity.cu
 *
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "plasticity_kernel.cu"

#include "assist.h"

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{
    bool if_quiet = true;
    unsigned int timer_compute = 0;
    unsigned int timer_memory = 0;
    int i;
    char input_fn[1024];
    char output_fn[1024];
    data_type * deviceBetaP = NULL, *deviceSigma = NULL;
    data_type * deviceFlux = NULL, *deviceVel = NULL;
    int width = N, height = N;

    int seed = 0;
    CUT_DEVICE_INIT(argc, argv);
    cutGetCmdLineArgumenti(argc, (const char **) argv, "seed", &seed);

#ifdef LENGTHSCALE
    float lengthscale = 0.;
    cutGetCmdLineArgumentf(argc, (const char **) argv, "lengthscale", &lengthscale);
    printf("Lengthscale: %.2f\n", lengthscale);
#endif

#ifdef LOADING
    printf("Loading\n");
#else
    printf("Relaxing\n");
#endif
    printf("Running seed: %d\n", seed);

    if_quiet = true; // If not display matrix contents

    //printf("Input matrix file name: %s\n", input_fn);

    // -----------------------------------------------------------------------
    // Setup host side
    // -----------------------------------------------------------------------

    printf("Setup host side environment and launch kernel:\n");

    // allocate host memory for matrices M and N
    printf("  Allocate host memory for matrices.\n");
#ifdef DIMENSION3
    printf("    N: %d x %d x %d x %d\n", N, N, N, NUM_COMP);
    unsigned int size = N * N * N * NUM_COMP;
    int breadth = N;
#else
    printf("    N: %d x %d x %d\n", N, N, NUM_COMP);
    unsigned int size = N * N * NUM_COMP;
    int breadth = 1;
#endif
    unsigned int mem_size = sizeof(data_type) * size;
    data_type* hostBetaP = (data_type*) malloc(mem_size);
    data_type* hostSigma = (data_type*) malloc(mem_size);
    data_type* hostFlux = (data_type*) malloc(mem_size);
    data_type* hostVel = (data_type*) malloc(mem_size);

    // Initialize the input matrices.
    printf("  Initialize the input matrices.\n");

    double time = 0.;
#ifndef LENGTHSCALE
    sprintf(output_fn, FILE_PREFIX FILE_PREFIX2  "cuda_" RUN_DESC "_%d_" PRECISION_STR "_%d_L%d.plas", N, seed, lambda);
#else
    sprintf(output_fn, FILE_PREFIX FILE_PREFIX2  "cuda_" RUN_DESC "_%d_" PRECISION_STR "_%d_L%d_l%.2f.plas", N, seed, lambda, lengthscale);
#endif

#ifdef CONTINUE_RUN
    FILE *test_fp = fopen(output_fn, "rb");
    if (test_fp != NULL) {
        fclose(test_fp);
        test_fp=NULL;
        // Saved file exists
        // Load previous state 
        data_type * matrix;
        matrix = ReadMatrixFileFunc(output_fn, 1, breadth*height*width*NUM_COMP+1, 1, if_quiet);
        time = (double)*matrix;
        printf(" Restarting from t=%f\n", time);
        matrix++;
        for(i = 0; i < size; i++)
            hostBetaP[i] = (data_type) matrix[i];
        matrix--;
        free(matrix);
    } else 
    {
#endif
    // Load from relaxed or initialized file for runs
#ifdef LOADING
        data_type * matrix;
        sprintf(input_fn, FILE_PREFIX FILE_PREFIX2 "cuda_" RELAX_RUN_DESC "_%d_" PRECISION_STR "_%d_L%d.plas", N, seed, lambda);
        matrix = ReadMatrixFileFunc(input_fn, width, breadth*height*NUM_COMP, 1, if_quiet);
#else
        double * matrix;
        //float * matrix;
#ifndef LENGTHSCALE
        sprintf(input_fn, FILE_PREFIX FILE_PREFIX2 "initial_%d_%d.mat", N, seed);
#else
        sprintf(input_fn, FILE_PREFIX FILE_PREFIX2 "initial_%d_%d_L%.2f.mat", N, seed, lengthscale);
#endif
        matrix = ReadDoubleMatrixFile(input_fn, width, breadth*height*NUM_COMP, 0, if_quiet);
#endif
        for(i = 0; i < size; i++)
            hostBetaP[i] = (data_type) matrix[i];
        free(matrix); matrix = NULL;
    }

    double timeInc = 0.01;

#ifdef LOADING
    double endTime = 3.00/LOADING_RATE;
#else
    double endTime = 20.00;
#endif
    FILE *data_fp = OpenFile(output_fn, 
#ifdef CONTINUE_RUN
        "ab",
#else
        "wb", 
#endif
        if_quiet);
#define XSTR(s) STR(s)
#define STR(s) #s
    //FILE *data_fp = OpenFile("cudaload_"XSTR(N)"_dp_L%d.plas", "wb", if_quiet);

    // ===================================================================
    //  Allocate device memory for the input matrices.
    //  Copy memory from the host memory to the device memory.
    // ===================================================================

    CUT_SAFE_CALL(cutCreateTimer(&timer_memory));
    CUT_SAFE_CALL(cutStartTimer(timer_memory));

    printf("  Allocate device memory.\n");

    CUDA_SAFE_CALL(cudaMalloc((void**) &deviceBetaP, mem_size));

    setupSystem();
    printf("  Copy host memory data to device.\n");
#ifdef DYNAMIC_NUCLEATION
    CUDA_SAFE_CALL(cudaMalloc((void**) &beta0dot, mem_size));
    CUDA_SAFE_CALL(cudaMemcpy(beta0dot, hostBetaP, mem_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(deviceBetaP, 0, mem_size));

{    d_dim_vector L;
    L.x = width;
    L.y = height;
#ifdef DIMENSION3
    L.z = breadth;
#endif

    data_type *sigma;
    CUDA_SAFE_CALL(cudaMalloc((void**) &sigma, mem_size)); 
    calculateSigma(beta0dot, sigma, L); 
    cudaThreadSynchronize();

    printf("max beta0 = %f\n", reduceMax(beta0dot, size));
    double max = 0.0; for (int i=0; i<size; i++){ if (hostBetaP[i] > max) max = hostBetaP[i];}
    printf("max host = %f\n", max);
    double sigmax = reduceMax(sigma, size);
    maxNucleationTimestep = 1.0/sqrt(fabs(sigmax));
    printf("sigmax  %f\n", sigmax);
    printf("maxNucleationTimestep = %f\n", maxNucleationTimestep);
    CUDA_SAFE_CALL(cudaFree(sigma));
}
#else
    CUDA_SAFE_CALL(cudaMemcpy(deviceBetaP, hostBetaP, mem_size,
        cudaMemcpyHostToDevice));
#endif

    printf("  Allocate device memory for results.\n");

    // FIXME - These lines maybe allocating unused memory
    /*
    CUDA_SAFE_CALL(cudaMalloc((void**) &deviceSigma, mem_size));
    cudaMemset(deviceSigma, 0, mem_size);
    CUDA_SAFE_CALL(cudaMalloc((void**) &deviceFlux, mem_size));
    cudaMemset(deviceFlux, 0, mem_size);
    CUDA_SAFE_CALL(cudaMalloc((void**) &deviceVel, mem_size));
    cudaMemset(deviceVel, 0, mem_size);
    */

    CUT_SAFE_CALL(cutStopTimer(timer_memory));

    // ================================================
    // Initialize the block and grid dimensions here
    // ================================================

    printf("  Executing the kernel...\n");

    // Start the timer_compute to calculate how much time we spent on it.
    CUT_SAFE_CALL(cutCreateTimer(&timer_compute));
    CUT_SAFE_CALL(cutStartTimer(timer_compute));


    d_dim_vector L;
    L.x = width;
    L.y = height;
#ifdef DIMENSION3
    L.z = breadth;
#endif

    // If this is the initial slice
#ifndef LAST_SHOT_ONLY
    if (time==0.)
        ContinueWriteMatrix( data_fp, hostBetaP, time, width, breadth*height*NUM_COMP, if_quiet); 
#endif

#ifndef DEBUG_TIMESTEPS
    while(time < endTime) {
        double intermediateTime;
#ifdef LOADING
        timeInc = 0.5;
#else
        if (time<=0.1) 
            timeInc = 0.01;
        else
            if (time <= 1.0)
                timeInc = 0.05;
            else
                if (time <= 5.0)
                    timeInc = 0.5;
                else
                    timeInc = 1.0;
#endif
        intermediateTime = time + timeInc;
        while(time < intermediateTime) {
            double timeStep = TVDstep(deviceBetaP, L, time, intermediateTime);
            printf("%le +%le\n", time, timeStep);
            time += timeStep;
        }
        cudaThreadSynchronize();
        cudaMemcpy(hostBetaP, deviceBetaP, mem_size, cudaMemcpyDeviceToHost);
#ifndef LAST_SHOT_ONLY
        ContinueWriteMatrix( data_fp, hostBetaP, time, width, breadth*height*NUM_COMP, if_quiet); 
#endif
    }
#ifdef LAST_SHOT_ONLY
    ContinueWriteMatrix( data_fp, hostBetaP, time, width, breadth*height*NUM_COMP, if_quiet);
#endif

#else
#ifndef SINGLE_STEP_DEBUG
    int count = 0;
    while(count++ < 10) {
        double intermediateTime = time+1.0;
        double timeStep = TVDstep(deviceBetaP, L, height, time, intermediateTime);
        printf("dbg %le +%le\n", time, timeStep);
        time += timeStep;

        cudaThreadSynchronize();
        cudaMemcpy(hostBetaP, deviceBetaP, mem_size, cudaMemcpyDeviceToHost);
        ContinueWriteMatrix( data_fp, hostBetaP, time, width, breadth*height*NUM_COMP, if_quiet); 
    }
#else
#ifdef DIMENSION3
#error
#endif
    dim3 grid(N/TILEX, N);
    dim3 tids(TILEX, 3, 3);
    data_type *sigma;
    CUDA_SAFE_CALL(cudaMalloc((void**) &sigma, sizeof(data_type)*breadth*width*height*NUM_SIG_COMP));
    data_type *rhs;
    CUDA_SAFE_CALL(cudaMalloc((void**) &rhs, sizeof(data_type)*breadth*width*height*NUM_COMP));
    data_type *velocity;
    CUDA_SAFE_CALL(cudaMalloc((void**) &velocity, sizeof(data_type)*breadth*width*height*NUM_COMP));

    calculateSigma(deviceBetaP, sigma, width, height);
    cudaThreadSynchronize();
    cudaMemcpy(hostBetaP, sigma, mem_size, cudaMemcpyDeviceToHost);
    ContinueWriteMatrix( data_fp, hostBetaP, time, width, breadth*height*NUM_COMP, if_quiet); 

    // calculate flux
    centralHJ<<<grid, tids>>>(deviceBetaP, sigma, rhs, velocity, L);

    cudaThreadSynchronize();
    cudaMemcpy(hostBetaP, rhs, mem_size, cudaMemcpyDeviceToHost);
    ContinueWriteMatrix( data_fp, hostBetaP, time, width, breadth*height*NUM_COMP, if_quiet); 
    
    cudaMemcpy(hostBetaP, velocity, mem_size/NUM_COMP, cudaMemcpyDeviceToHost);
    ContinueWriteMatrix( data_fp, hostBetaP, time, width, breadth*height*NUM_COMP, if_quiet); 
#endif
#endif
    // Make sure all threads have finished their jobs
    // before we stop the timer_compute.
    cudaThreadSynchronize();

    fclose( data_fp );
    
    // Stop the timer_compute
    CUT_SAFE_CALL(cutStopTimer(timer_compute));

    // check if kernel execution generated an error
    ERROR_CHECK
    CUT_CHECK_ERROR("Kernel execution failed");

    // ===================================================================
    // Copy the results back from the host
    // ===================================================================

    printf("  Copy result from device to host.\n");

    CUT_SAFE_CALL(cutStartTimer(timer_memory));
    //cudaMemcpy(hostSigma, deviceSigma, mem_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(hostFlux, deviceFlux, mem_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(hostVel, deviceVel, mem_size, cudaMemcpyDeviceToHost);
    CUT_SAFE_CALL(cutStopTimer(timer_memory));

    // ================================================
    // Show timing information
    // ================================================

    printf("  GPU memory access time: %f (ms)\n",
        cutGetTimerValue(timer_memory));
    printf("  GPU computation time  : %f (ms)\n",
        cutGetTimerValue(timer_compute));
    printf("  GPU processing time   : %f (ms)\n",
        cutGetTimerValue(timer_compute) + cutGetTimerValue(timer_memory));
    CUT_SAFE_CALL(cutDeleteTimer(timer_memory));
    CUT_SAFE_CALL(cutDeleteTimer(timer_compute));

    //WriteMatrixFile("velocity.mat", hostVel, width, height, if_quiet);
    //WriteMatrixFile("rhs.mat", hostFlux, width, NUM_COMP*height, if_quiet); 
#if 0 
    for(i = 0; i < NUM_SIG_COMP; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < width; k++)
                fprintf(stdout, "%lf ", hostSigma[(i*height+j)*width+k]);
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "\n");
    }
#endif
    // clean up memory
    free(hostBetaP); free(hostSigma);
    free(hostFlux); free(hostVel);

    // ===================================================================
    // Free the device memory
    // ===================================================================

    CUDA_SAFE_CALL(cudaFree(deviceBetaP));
    //CUDA_SAFE_CALL(cudaFree(deviceSigma));
    //CUDA_SAFE_CALL(cudaFree(deviceFlux));
    //CUDA_SAFE_CALL(cudaFree(deviceVel));
}


