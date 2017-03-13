/******************************************************************************
* Project				: Vehicle Routing Problem Algorithm on Nvidia GTX 960M GPU
* File name				: vrp.cu
* Author				: Prasad Pandit & Radhika Mandlekar
* Date Modified			: 03/11/2017
* Contact				: prasad@pdx.edu & radhika@pdx.edu
* Description			:
******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <device_launch_parameters.h>
#include <device_functions.h>

#include <helper_cuda.h>

/*
* CUDA Kernel Device code
*/
__global__ void
savingsCalc(int* costMatrix, int* savingsMatrix, int rows, int columns)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	if (i<rows && j<rows) {
		//savingsMatrix[i][j] = (i!=j && j>i) ? costMatrix[0][i] + costMatrix[0][j] -	costMatrix[i][j] : 0;
		*(savingsMatrix + i*columns + j) = (i != j && j > i) ? *(costMatrix + 0 + i) + *(costMatrix + 0 + j) - *(costMatrix + i*columns + j) : 0;
		}
}

__global__ void
savingsSort() {
}

struct savings_info
{
	int startNode;
	int endNode;
	int savingsBetweenNodes;
};

struct route
{
	int nodes_in_route[1024];
	int nodesAdded;
};
struct demandInfo
{
	int node;
	int demand;
};

struct keyVal
{
	int key;
	int val;
	int routeIndex;
	int indexOfnodeInRouteInResultArray;
};

__global__ void
sortSavings(struct savings_info* records, int count) {
	int i = threadIdx.x;
	
	//int j,l;

	//for (l = 1; l<count; l++)
	//{
	//	for (j = 0; j< count - l ; j++)
	//	{
	//		if (records[i].savingsBetweenNodes < records[i + 1].savingsBetweenNodes)
	//		{
	//			struct savings_info temp = records[i];
	//			records[i] = records[i + 1];
	//			records[i + 1] = temp;
	//		}
	//	}
	//}


	int l;

	if (count % 2 == 0)
		l = count / 2;
	else
		l = (count / 2) + 1;

	for (int j = 0; j<l; j++)
	{
		if ((!(i & 1)) && (i<(count - 1)))  //even phase
		{
			if (records[i].savingsBetweenNodes < records[i + 1].savingsBetweenNodes) {
				struct savings_info temp = records[i];
				records[i] = records[i + 1];
				records[i + 1] = temp;
			}
				//intswap(records[i], records[i + 1]);
		}
		__syncthreads();

		if ((i & 1) && (i<(count - 1)))     //odd phase
		{
			if (records[i].savingsBetweenNodes < records[i + 1].savingsBetweenNodes) {
				struct savings_info temp = records[i];
				records[i] = records[i + 1];
				records[i + 1] = temp;
			}
		}
		__syncthreads();
		
	}//for
}

savings_info* sort_savings_structure(struct savings_info* records, int n) {
	int j, i;

	for (i = 1; i<n; i++)
	{
		for (j = 0; j<n - i; j++)
		{
			if (records[j].savingsBetweenNodes < records[j + 1].savingsBetweenNodes)
			{
				struct savings_info temp = records[j];
				records[j] = records[j + 1];
				records[j + 1] = temp;
			}
		}
	}
	return records;
}



/**
* Host main routine
*/
int main(void)
{
	// Internal Host Variables
	char lineInFile[1024];

	// Read VRP Parameters for initializations
	FILE *file_parms = fopen("C:/Users/prasa/Projects/CUDA/VRP/vrp_parameters.txt", "r");
	int rows = 0, columns = 0;

	if (file_parms)
	{

		fgets(lineInFile, sizeof(lineInFile), file_parms);
		//printf("lineInFile: %s", lineInFile);
		sscanf(lineInFile, "cost_matrix_size rows %d,columns %d", &rows, &columns);
		printf("Rows = %d, Columns = %d, Total Elements = %d", rows, columns, rows*columns);
		fclose(file_parms);
	}

	// Initialize the host costMatrix_h

	int *costMatrix_h = (int *)malloc(rows * columns * sizeof(int));

	// Get cost matrix values from csv and initialize 2D matrix

	FILE* cost_matrix_csv = fopen("C:/Users/prasa/Projects/CUDA/VRP/vrp_cost_matrix.csv", "r");

	int row_count = 0, column_count = 0;

	while (fgets(lineInFile, 1024, cost_matrix_csv))
	{
		const char* delimiter = ",";
		char *each_char =strtok(lineInFile, delimiter);
		while (each_char != NULL)
		{
			if (NULL != each_char) {
				int i;
				sscanf(each_char, "%d", &i);
				*(costMatrix_h + row_count*columns + column_count) = i;
				column_count++;
			}
			each_char = strtok(NULL, delimiter);

		}
		column_count = 0;
		row_count++;
	}
	fclose(cost_matrix_csv);
	printf("Initialized cost matrix : Done\n");

	printf("\nPrinting the costMatrix_h below:\n\n");
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < columns; ++j) {
			printf("%2d\t", *(costMatrix_h + i*columns + j));
		}
		printf("\n");
	}

	// Get demand parameter values from csv and initialize 1D array

	struct demandInfo *demandParam_h = (demandInfo *)malloc((rows - 1) * sizeof(demandInfo));

	FILE* demand_parameters_csv = fopen("C:/Users/prasa/Projects/CUDA/VRP/vrp_demand_parameters.csv", "r");

	int node_count = 0;

	while (fgets(lineInFile, 1024, demand_parameters_csv))
	{
		sscanf(lineInFile, "%d,%d", &demandParam_h[node_count].node, &demandParam_h[node_count].demand);
		node_count++;
	}
	fclose(demand_parameters_csv);
	printf("Initialized demand array : Done\n");

	printf("\nPrinting the demandParam_h below:\n\n");
	for (int i = 0; i < node_count; ++i)
	{
		printf("%2d\t,%2d\n", demandParam_h[i].node, demandParam_h[i].demand);
	}
	
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	size_t size = rows * columns * sizeof(int);
	long total_size = rows * columns * sizeof(int);
	int *savingsMatrix_h = (int*) malloc(rows * columns * sizeof(int));
	
	// Allocate the device input vector A
	int *costMatrix_d = (int*) malloc(rows * columns * sizeof(int));
	
	//(void **)devptr pointer to allocated device memory 
	err = cudaMalloc((void **)&costMatrix_d, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	// Allocate the device output vector C
	int *savingsMatrix_d = (int*)malloc(rows * columns * sizeof(int));
	err = cudaMalloc((void **)&savingsMatrix_d, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	//( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(costMatrix_d, costMatrix_h, size, cudaMemcpyHostToDevice);
	
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	// Launch the Vector Add CUDA Kernel
	dim3 threadsPerBlock(rows, columns);
	int blocksPerGrid = 1;//((numNodes*numNodes) + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks \n", blocksPerGrid);
	
	savingsCalc <<< blocksPerGrid, threadsPerBlock >>>((int *)costMatrix_d,(int *) savingsMatrix_d, rows,columns);
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
	
	struct savings_info *records_h = (savings_info*)malloc(rows * columns * sizeof(savings_info));

	int count = 0;
	// Verify that the result vector is correct
	printf("\nPrinting the savingsMatrix_h below:\n\n");
	for (int i = 1; i < rows-1; ++i)
	{
		for (int j = 2; j < columns; ++j) {
			if (i < j) {
			printf("%2d\t", *(savingsMatrix_h + i*columns + j));
			records_h[count].startNode = i;
			records_h[count].endNode = j;
			records_h[count].savingsBetweenNodes = *(savingsMatrix_h + i*columns + j);
			count++;
			}
			
		}
		printf("\n");
	}

	/*Serial Sort*/
	/*
	struct savings_info *sortedSavingsRecords;
	sortedSavingsRecords= sort_savings_structure(records,count);

	//printngs sorted Savings info structure
	for (int i = 0; i < count; i++) {
		printf("Sorted Record %2d = {%2d\t, %2d\t,%2d\t} ", i, sortedSavingsRecords[i].startNode, sortedSavingsRecords[i].endNode, sortedSavingsRecords[i].savingsBetweenNodes);
		printf("\n");
	}
	printf("\n");
	*/

	//PArallel Sorting 
	size_t size_records = rows * columns * sizeof(savings_info);
	struct savings_info *records_d = (savings_info*)malloc(rows * columns * sizeof(savings_info));
	err = cudaMalloc((void **)&records_d, size_records);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err = cudaMemcpy(records_d, records_h, size_records, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Launch the Vector Add CUDA Kernel
	dim3 threadsPerBlock_savingsSort(count);
	blocksPerGrid = 1;//((numNodes*numNodes) + threadsPerBlock - 1) / threadsPerBlock;
	printf("RADHIKA : CUDA kernel launch with %d blocks \n", blocksPerGrid);

	//savingsCalc << < blocksPerGrid, threadsPerBlock >> >((int *)costMatrix_d, (int *)savingsMatrix_d, rows, columns);
	sortSavings <<< blocksPerGrid, threadsPerBlock_savingsSort >>>((savings_info *)records_d, count);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch savingsCalc kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
	
	err = cudaMemcpy(records_h, records_d, size_records, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < count; i++) {
		printf("Sorted Record %2d = {%2d\t, %2d\t,%2d\t} ", i, records_h[i].startNode, records_h[i].endNode, records_h[i].savingsBetweenNodes);
		printf("\n");
	}

	//****************************************************************************//
	/*************MAIN AALGORITHM : CLARKE AND WRITE ******************************/
	//****************************************************************************//

	// Result dictionary to capture node visit updates
	int vehicleCapacity = 40;
	int numOfVehicles = 5;
	struct keyVal *resultDict_h = (keyVal *)malloc(node_count * sizeof(keyVal));

	for (int i = 0; i < node_count; i++)
	{
		resultDict_h[i].key = demandParam_h[i].node;
		resultDict_h[i].val = 0;
	}

	// Result list to store final paths
	struct route *resultList_h = (route *)malloc(numOfVehicles * sizeof(route));
	
	int main_algo_i = 0;
	int exist_i = 0;
	int exist_j = 0;
	int	index_of_tuple_for_i = 0;
	int	index_of_tuple_for_j = 0;
	int nodesProcessed = 0;
	int routesAdded = 0;

	printf("\nNo of elements of  records_h :%d \n\n" , count);
	
	for (int z = 0; z < count; z++) {
		int edge_i = records_h[z].startNode;
		int edge_j = records_h[z].endNode;
		
		int demandParam_h_edge_i;
		int demandParam_h_edge_j;
		int demandParam_h_edge_i_dem;
		int demandParam_h_edge_j_dem;
		for (int demand_count = 0; demand_count < node_count; demand_count++) {
			if (demandParam_h[demand_count].node == edge_i ) {
				demandParam_h_edge_i = demandParam_h[demand_count].node;
				demandParam_h_edge_i_dem = demandParam_h[demand_count].demand;
				printf("\n\nDemand of %d is %d\n", edge_i, demandParam_h_edge_i_dem);
			}else if(demandParam_h[demand_count].node == edge_j) {
				demandParam_h_edge_j = demandParam_h[demand_count].node;
				demandParam_h_edge_j_dem = demandParam_h[demand_count].demand;
				printf("\nDemand of %d is %d\n", edge_j, demandParam_h_edge_j_dem);
			}
		}
		printf("\nDemands : %d and %d for Edges : %d and %d \n", demandParam_h_edge_i_dem, demandParam_h_edge_j_dem, edge_i, edge_j);
		
		if (nodesProcessed != 0) {
			if (demandParam_h_edge_i_dem + demandParam_h_edge_j_dem <= vehicleCapacity) {
				printf("Iteration No.: %d for %d , %d\n", z, edge_i, edge_j);
				//printf("1. Capacity Constraints Not Violated : %d \n", demandParam_h_edge_i_dem + demandParam_h_edge_j_dem);
				int numberOfRoutes = routesAdded;
				
				if (resultDict_h[edge_i-1].val == 1 && resultDict_h[edge_j-1].val == 0) 
				{
					int indexOfRoute = resultDict_h[edge_i-1].routeIndex;
					int numberOfNodesInRoute = resultList_h[indexOfRoute].nodesAdded;
					int total_demand = 0;
					total_demand += demandParam_h_edge_j_dem;
					for (int temp_i = 0; temp_i < numberOfNodesInRoute; temp_i++) 
					{
						total_demand += demandParam_h[resultList_h[indexOfRoute].nodes_in_route[temp_i] - 1].demand;
					}
					printf("Total Demand: %d\n", total_demand);
					if (total_demand <= vehicleCapacity) 
					{
						if (resultDict_h[edge_i-1].indexOfnodeInRouteInResultArray == 0 || resultDict_h[edge_i-1].indexOfnodeInRouteInResultArray == (resultList_h[indexOfRoute].nodesAdded - 1))
						{
							resultList_h[indexOfRoute].nodes_in_route[numberOfNodesInRoute] = edge_j;
							resultList_h[indexOfRoute].nodesAdded += 1;
							resultDict_h[edge_j-1].val = 1;
							resultDict_h[edge_j-1].routeIndex = indexOfRoute;
							resultDict_h[edge_j-1].indexOfnodeInRouteInResultArray = numberOfNodesInRoute;
							nodesProcessed += 1;
							printf("i. Added Node %d after %d\n", edge_j, edge_i);
							//routesAdded += 1;
						}
						else 
						{
							printf("i. Can't add the %d node as %d is intermediate node \n", edge_j, edge_i);
						}									
					}
					else 
					{
						printf("i. Capacity exceeding for nodes %d and %d \n" , edge_i, edge_j);
					}
				}
				else if (resultDict_h[edge_i-1].val == 0 && resultDict_h[edge_j-1].val == 1) 
				{
					int indexOfRoute = resultDict_h[edge_j-1].routeIndex;
					int numberOfNodesInRoute = resultList_h[indexOfRoute].nodesAdded;
					int total_demand = 0;
					total_demand += demandParam_h_edge_i_dem;
					for (int temp_i = 0; temp_i < numberOfNodesInRoute; temp_i++)
					{
						total_demand += demandParam_h[resultList_h[indexOfRoute].nodes_in_route[temp_i] - 1].demand;
					}
					printf("Total Demand: %d\n", total_demand);
					if (total_demand <= vehicleCapacity)
					{
						if (resultDict_h[edge_j-1].indexOfnodeInRouteInResultArray == 0 || resultDict_h[edge_j-1].indexOfnodeInRouteInResultArray == (resultList_h[indexOfRoute].nodesAdded - 1))
						{
							resultList_h[indexOfRoute].nodes_in_route[numberOfNodesInRoute] = edge_i;
							resultList_h[indexOfRoute].nodesAdded += 1;
							resultDict_h[edge_i-1].val = 1;
							resultDict_h[edge_i-1].routeIndex = indexOfRoute;
							resultDict_h[edge_i-1].indexOfnodeInRouteInResultArray = numberOfNodesInRoute;
							nodesProcessed += 1;
							printf("j. Added Node %d after %d\n", edge_i, edge_j);
							//routesAdded += 1;
						}
						else
						{
							printf("j.Can't add the %d node as %d is intermediate node \n", edge_i, edge_j);
						}
					}
					else
					{
						printf("j. Capacity exceeding for nodes %d and %d \n", edge_i, edge_j);
					}
				}
				else if (resultDict_h[edge_i-1].val == 0 && resultDict_h[edge_j-1].val == 0)
				{
					resultList_h[routesAdded].nodes_in_route[0] = edge_i;
					resultList_h[routesAdded].nodes_in_route[1] = edge_j;
					resultList_h[routesAdded].nodesAdded = 2;
					resultDict_h[edge_i-1].val = 1;
					resultDict_h[edge_j-1].val = 1;
					resultDict_h[edge_i-1].routeIndex = routesAdded;
					resultDict_h[edge_j-1].routeIndex = routesAdded;
					resultDict_h[edge_i-1].indexOfnodeInRouteInResultArray = 0;
					resultDict_h[edge_j-1].indexOfnodeInRouteInResultArray = 1;
					nodesProcessed += 2;
					routesAdded += 1;
					printf("Added both nodes %d and %d\n", edge_i, edge_j);
					printf("Route Added : numberOfRoutes = %d \n", routesAdded);
				}
				else
				{
					printf("Nodes %d and %d are already processed\n", edge_i, edge_j);
				}				
			}
			else 
			{
				printf("Iteration No.: %d for %d , %d \n", z, edge_i, edge_j);
				printf("Capacity Constraints Violated : %d \n ", demandParam_h_edge_i_dem + demandParam_h_edge_j_dem);
			}
		}
		else 
		{
			if (demandParam_h_edge_i_dem + demandParam_h_edge_j_dem <= vehicleCapacity) 
			{
				printf("Iteration No.: %d for %d , %d \n", z, edge_i, edge_j);
				printf("2. Capacity Constraints Not Violated : %d \n", demandParam_h_edge_i_dem + demandParam_h_edge_j_dem);

				resultList_h[routesAdded].nodes_in_route[0]  = edge_i;
				resultList_h[routesAdded].nodes_in_route[1]  = edge_j;
				resultList_h[routesAdded].nodesAdded = 2;
				resultDict_h[edge_i-1].val = 1;
				resultDict_h[edge_j-1].val = 1;
				resultDict_h[edge_i-1].routeIndex = routesAdded;
				resultDict_h[edge_j-1].routeIndex = routesAdded;
				resultDict_h[edge_i-1].indexOfnodeInRouteInResultArray = 0;
				resultDict_h[edge_j-1].indexOfnodeInRouteInResultArray = 1;
				nodesProcessed += 2;
				routesAdded += 1;
				printf(" Route Added numberOfRoutes = %d %d,%d\n ", routesAdded, edge_i, edge_j);
			}
		}
	}//End of for_count 
	
	printf("resultList_h \n ");
	for (int z = 0; z < routesAdded; z++) 
	{
		struct route temproute = resultList_h[z];
		
		 printf("\nRoute: %d\n", z);
		 printf("NodesAdded: %d\n", temproute.nodesAdded);

		 for (int i = 0; i < temproute.nodesAdded; i++)
		 {
			 	 printf(" %d \t", temproute.nodes_in_route[i]);
		 }
		 printf("\n");
		 	
	}
	
	//****************************************************************************//
	//***********************End of Main Algorithm********************************//
	//****************************************************************************//

	// Free device global memory
	err = cudaFree(costMatrix_d);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(savingsMatrix_d);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(costMatrix_h);
	free(savingsMatrix_h);
	free(records_h);
	
	printf("Done\n");
	scanf("");
	return 0;
}
