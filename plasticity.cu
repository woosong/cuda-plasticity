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
    printf("    N: %d x %d x 9\n", N, N);
    unsigned int size = N * N * 9;
    unsigned int mem_size = sizeof(data_type) * size;
    data_type* hostBetaP = (data_type*) malloc(mem_size);
    data_type* hostSigma = (data_type*) malloc(mem_size);
    data_type* hostFlux = (data_type*) malloc(mem_size);
    data_type* hostVel = (data_type*) malloc(mem_size);

    // Initialize the input matrices.
    printf("  Initialize the input matrices.\n");
    /*
    unsigned int * matrix;
    matrix = ReadMatrixFile(input_fn, width, height*9, 0, if_quiet);
    for(i = 0; i < 9*width*height; i++)
        hostBetaP[i] = (data_type) matrix[i];
    free(matrix); matrix = NULL;
    */
#if 0
    for(i = 0; i < 9*width*height; i++)
        hostBetaP[i] = 0.;
    for(i = 0; i < width*height; i++)
        hostBetaP[i] = sin((i%width)*2.*M_PI/width)*sin((i/width)*2.*M_PI/height);
    hostBetaP[0] = 1;
#else
#ifdef LOADING
    data_type * matrix;
    sprintf(input_fn, "cuda_" RELAX_RUN_DESC "_%d_" PRECISION_STR "_%d_L%d.plas", N, seed, lambda);
#ifdef DOUBLE
    matrix = ReadDoubleMatrixFile(input_fn, width, height*9, 1, if_quiet);
#else
    matrix = ReadMatrixFile(input_fn, width, height*9, 1, if_quiet);
#endif
#else
    double * matrix;
    //float * matrix;
    sprintf(input_fn, "initial_%d_%d.mat", N, seed);
    matrix = ReadDoubleMatrixFile(input_fn, width, height*9, 0, if_quiet);
#endif
    for(i = 0; i < 9*width*height; i++)
        hostBetaP[i] = (data_type) matrix[i];
    free(matrix); matrix = NULL;
#endif
#if 0
    for(i = 0; i < 9; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < width; k++)
                fprintf(stdout, "%f ", hostBetaP[(i*height+j)*width+k]);
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "\n");
    }
#endif
    // ===================================================================
    //  Allocate device memory for the input matrices.
    //  Copy memory from the host memory to the device memory.
    // ===================================================================

    CUT_SAFE_CALL(cutCreateTimer(&timer_memory));
    CUT_SAFE_CALL(cutStartTimer(timer_memory));

    printf("  Allocate device memory.\n");

    CUDA_SAFE_CALL(cudaMalloc((void**) &deviceBetaP, mem_size));

    printf("  Copy host memory data to device.\n");

    CUDA_SAFE_CALL(cudaMemcpy(deviceBetaP, hostBetaP, mem_size,
        cudaMemcpyHostToDevice));

    printf("  Allocate device memory for results.\n");

    CUDA_SAFE_CALL(cudaMalloc((void**) &deviceSigma, mem_size));
    cudaMemset(deviceSigma, 0, mem_size);
    CUDA_SAFE_CALL(cudaMalloc((void**) &deviceFlux, mem_size));
    cudaMemset(deviceFlux, 0, mem_size);
    CUDA_SAFE_CALL(cudaMalloc((void**) &deviceVel, mem_size));
    cudaMemset(deviceVel, 0, mem_size);

    CUT_SAFE_CALL(cutStopTimer(timer_memory));

    // ================================================
    // Initialize the block and grid dimensions here
    // ================================================

    printf("  Executing the kernel...\n");

    // Start the timer_compute to calculate how much time we spent on it.
    CUT_SAFE_CALL(cutCreateTimer(&timer_compute));
    CUT_SAFE_CALL(cutStartTimer(timer_compute));

    setupSystem();

    // Invoke the CUDA kernel here
    //calculateSigma(deviceBetaP, deviceSigma, height, width);
    //calculateSigma(deviceBetaP, deviceSigma, width, height);
    //printf("%le\n", (double)reduceMax(deviceSigma, 9*width*height));

    //calculateFlux(deviceBetaP, deviceFlux, deviceVel, width, height);
    //printf("%le\n", (double)reduceMax(deviceVel, width*height));

    //runTVD(deviceBetaP, width, height, 0., 5.00);
    double time = 0.;
    double timeInc = 0.01;

#ifdef LOADING
    double endTime = 4.00/LOADING_RATE;
#else
    double endTime = 20.00;
#endif
    sprintf(output_fn, "cuda_" RUN_DESC "_%d_" PRECISION_STR "_%d_L%d.plas", N, seed, lambda);
    FILE *data_fp = OpenFile(output_fn, "wb", if_quiet);
#define XSTR(s) STR(s)
#define STR(s) #s
    //FILE *data_fp = OpenFile("cudaload_"XSTR(N)"_dp_L%d.plas", "wb", if_quiet);

    ContinueWriteMatrix( data_fp, hostBetaP, time, width, height*9, if_quiet); 

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
            double timeStep = TVD3rd(deviceBetaP, width, height, time, intermediateTime);
            printf("%le +%le\n", time, timeStep);
            time += timeStep;
        }
        cudaThreadSynchronize();
        cudaMemcpy(hostBetaP, deviceBetaP, mem_size, cudaMemcpyDeviceToHost);
        ContinueWriteMatrix( data_fp, hostBetaP, time, width, height*9, if_quiet); 
    }
#else
#ifndef SINGLE_STEP_DEBUG
    int count = 0;
    while(count++ < 10) {
        double intermediateTime = time+1.0;
        double timeStep = TVD3rd(deviceBetaP, width, height, time, intermediateTime);
        printf("dbg %le +%le\n", time, timeStep);
        time += timeStep;

        cudaThreadSynchronize();
        cudaMemcpy(hostBetaP, deviceBetaP, mem_size, cudaMemcpyDeviceToHost);
        ContinueWriteMatrix( data_fp, hostBetaP, time, width, height*9, if_quiet); 
    }
#else
    dim3 grid(N/TILEX, N);
    dim3 tids(TILEX, 3, 3);
    data_type *sigma;
    CUDA_SAFE_CALL(cudaMalloc((void**) &sigma, sizeof(data_type)*width*height*9));
    data_type *rhs;
    CUDA_SAFE_CALL(cudaMalloc((void**) &rhs, sizeof(data_type)*width*height*9));
    data_type *velocity;
    CUDA_SAFE_CALL(cudaMalloc((void**) &velocity, sizeof(data_type)*width*height*9));

    calculateSigma(deviceBetaP, sigma, width, height);
    cudaThreadSynchronize();
    cudaMemcpy(hostBetaP, sigma, mem_size, cudaMemcpyDeviceToHost);
    ContinueWriteMatrix( data_fp, hostBetaP, time, width, height*9, if_quiet); 

    // calculate flux
    centralHJ<<<grid, tids>>>(deviceBetaP, sigma, rhs, velocity, width, height);

    cudaThreadSynchronize();
    cudaMemcpy(hostBetaP, rhs, mem_size, cudaMemcpyDeviceToHost);
    ContinueWriteMatrix( data_fp, hostBetaP, time, width, height*9, if_quiet); 
    
    cudaMemcpy(hostBetaP, velocity, mem_size/9, cudaMemcpyDeviceToHost);
    ContinueWriteMatrix( data_fp, hostBetaP, time, width, height*9, if_quiet); 
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
    cudaMemcpy(hostSigma, deviceSigma, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostFlux, deviceFlux, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostVel, deviceVel, mem_size, cudaMemcpyDeviceToHost);
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
    //WriteMatrixFile("rhs.mat", hostFlux, width, 9*height, if_quiet); 
#if 0 
    for(i = 0; i < 9; i++) {
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
    CUDA_SAFE_CALL(cudaFree(deviceSigma));
    CUDA_SAFE_CALL(cudaFree(deviceFlux));
    CUDA_SAFE_CALL(cudaFree(deviceVel));
}


