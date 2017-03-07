/**
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
* Vector addition: C = A + B.
*
* This sample is a very basic sample that implements element by element
* vector addition. It is the same as the sample illustrating Chapter 2
* of the programming guide with some additions like error checking.
*/

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <helper_cuda.h>
/**
* CUDA Kernel Device code
*
* Computes the vector addition of A and B into C. The 3 vectors have the same
* number of elements numNodes.
*/
__global__ void
savingsCalc(int costMatrix[][10], int savingsMatrix[][10], int numNodes)
{
	int i = threadIdx.x;
	int j = threadIdx.y;

	if (i<numNodes && j<numNodes) {
		savingsMatrix[i][j] = costMatrix[i][j];
	}

}

/**
* Host main routine
*/
int
main(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	const int numNodes = 10;
	size_t size = numNodes * numNodes * sizeof(int);
	long total_size = numNodes * numNodes * sizeof(int);
	printf("Total Size of Elements: %d", total_size);

	// // Allocate the host input vector A
	// int *costMatrix_h = (int *)malloc(size);
	//
	// // Allocate the host output vector C
	// int *savingsMatrix_h = (int *)malloc(size);

	// // Verify that allocations succeeded
	// if (costMatrix_h == NULL || savingsMatrix_h == NULL)
	// {
	//     fprintf(stderr, "Failed to allocate host vectors!\n");
	//     exit(EXIT_FAILURE);
	// }

	// Initialize the host costMatrix_h
	int costMatrix_h[numNodes][numNodes] = {
		{ 0,12,11,7,10,10,9,8,6,12 },
		{ 12,0,8,5,9,12,14,16,17,22 },
		{ 11,8,0,9,15,17,8,18,14,22 },
		{ 7,5,9,0,7,9,11,12,12,17 },
		{ 10,9,15,7,0,3,17,7,15,18 },
		{ 10,12,17,9,3,0,18,6,15,15 },
		{ 9,14,8,11,17,18,0,16,8,16 },
		{ 8,16,18,12,7,6,16,0,11,11 },
		{ 6,17,14,12,15,15,8,11,0,10 },
		{ 12,22,22,17,18,15,16,11,10,0 }
	};

	int savingsMatrix_h[numNodes][numNodes] = { 0 };

	// Allocate the device input vector A

	int (*costMatrix_d)[numNodes] = NULL;
	err = cudaMalloc((void **)&costMatrix_d, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	int (*savingsMatrix_d)[numNodes] = NULL;
	err = cudaMalloc((void **)&savingsMatrix_d, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(costMatrix_d, costMatrix_h, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	dim3 threadsPerBlock(numNodes, numNodes);
	int blocksPerGrid = 1;//((numNodes*numNodes) + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks \n", blocksPerGrid);
	savingsCalc <<<blocksPerGrid, threadsPerBlock >>>(costMatrix_d, savingsMatrix_d, numNodes);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch savingsCalc kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(savingsMatrix_h, savingsMatrix_d, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct
	printf("\nPrinting the savingsMatrix_h below:\n\n");
	for (int i = 0; i < numNodes; ++i)
	{
		for (int j = 0; j < numNodes; ++j) {
			printf("%2d\t", savingsMatrix_h[i][j]);
			if (costMatrix_h[i][j] != savingsMatrix_h[i][j]) {
				printf("Matrix match failed...!!!!!!! Dhungya..!!!!\n");
				exit(EXIT_FAILURE);
			}
		}
		printf("\n");
	}

	printf("Test PASSED\n");

	//// Free device global memory
	//err = cudaFree(costMatrix_d);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	//err = cudaFree(savingsMatrix_d);

	//if (err != cudaSuccess)
	//{
	//	fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
	//	exit(EXIT_FAILURE);
	//}

	//// Free host memory
	//free(costMatrix_h);
	//free(savingsMatrix_h);

	printf("Done\n");
	scanf("");
	return 0;
}
