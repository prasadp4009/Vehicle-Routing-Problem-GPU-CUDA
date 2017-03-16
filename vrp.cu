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
#include <ctype.h>
#include <time.h>


// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <helper_cuda.h>

// GPU Capacity

// Max Threads per block
#define MaxThreadsPerBlock 1024
#define MaxThreadx 32
#define MaxThready 32

/****************************************************************************/
struct nodeInfo {
	int node;
	int xCoOrd;
	int yCoOrd;
	int demand;
};

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
/*****************************************************************************/
/*
* CUDA Kernel Device code
*/
__global__ void
savingsCalc(int* costMatrix, int* savingsMatrix, int rows, int columns)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	//int tid = i + j * blockDim.x * gridDim.x;
	//int N = rows * columns;
	if (y < rows && x < columns) {
		//savingsMatrix[i][j] = (i!=j && j>i) ? costMatrix[0][i] + costMatrix[0][j] -	costMatrix[i][j] : 0;
		
		int valOfx = costMatrix[x];
		int valOfy = costMatrix[y];
		int valOfxy = costMatrix[x + y * rows];
		*(savingsMatrix + x + y * rows) = (x != y && x > y) ? valOfx + valOfy - valOfxy : 0;
		
		//*(savingsMatrix + i*columns + j) = (i != j && j > i) ? *(costMatrix + 0 + i) + *(costMatrix + 0 + j) - *(costMatrix + i*columns + j) : 0;
		//*(savingsMatrix + tid ) = (i != j && j > i) ? *(costMatrix + 0 + i) + *(costMatrix + 0 + j) - *(costMatrix + tid) : 0;
		}
}

__global__ void
sortSavings(struct savings_info* records, int count) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int l;
	if (count % 2 == 0)
		l = count / 2;
	else
		l = (count / 2) + 1;

	for (int j = 0; j<l; j++)
	{
		if ((!(i & 1)) && (i<(count - 1)))  //even phase
		{
			if (records[i].savingsBetweenNodes < records[i + 1].savingsBetweenNodes)
			{
				struct savings_info temp = records[i];
				records[i] = records[i + 1];
				records[i + 1] = temp;
			}
		}
		__syncthreads();

		if ((i & 1) && (i<(count - 1)))     //odd phase
		{
			if (records[i].savingsBetweenNodes < records[i + 1].savingsBetweenNodes)
			{
				struct savings_info temp = records[i];
				records[i] = records[i + 1];
				records[i + 1] = temp;
			}
		}
		__syncthreads();

	}//for
}

__global__ void
getCostMatrix(struct nodeInfo* localNodeInfo, int *localCostMatrix, int rows, int columns)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < rows & y < columns) {
		struct nodeInfo tempNode0 = localNodeInfo[x];
		struct nodeInfo tempNode1 = localNodeInfo[y];
		*(localCostMatrix + x*columns + y) = (x == y) ? 0 : __float2uint_ru(__fsqrt_ru((float)(((tempNode0.xCoOrd-tempNode1.xCoOrd)*(tempNode0.xCoOrd - tempNode1.xCoOrd))+((tempNode0.yCoOrd - tempNode1.yCoOrd)*(tempNode0.yCoOrd - tempNode1.yCoOrd)))));
	}
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

/**************************** Internal C Functions **************************/
int isNotAlpha(char *str)
{
	int len = strlen(str);
	for(int i = 0; i < len; i++)
	{
		if(isalpha(str[i]))
			return 0;
	}
	return 1;
}
/****************************************************************************/

/**
* Host main routine
*/
int main(int argc, char *argv[])
{
	// Internal Host Variables
	char lineInFile[1024];
	char fileName[1024];
	
	if (argc != 2) /* argc should be 2 for correct execution */
	{
		printf("Filename Missing");
		return 1;
	}
	else {
		sscanf(argv[1],"%[^\t\n]",fileName);
		printf("File to process: %s", &fileName);
	}
	FILE* vrp_input_file = fopen(fileName,"r");
	if (vrp_input_file == NULL) {
		return 0;
	}
	char	nameOfDataset[1024];
	char	commentOfDataset[1024];
	char	typeOfDataset[1024];
	char	edgeTypeOfDataset[1024];
	int		capacitOfVehicle = 0;
	int		totalNodes = 0;

	int	got_name		= 0,
		got_dimension	= 0,
		got_capacity	= 0,
		got_co_ord		= 0,
		got_demand_val	= 0;

	if(NULL == vrp_input_file)
	{
		printf("\n Failed to open file \n");
		return 1;
	}

	if(fgets(lineInFile, 1024, vrp_input_file))
	{
		if((strcmp(lineInFile, "NAME")!= 0) && !(got_name))
		{
			//printf("\nReceived Line: %s", lineInFile);
			sscanf(lineInFile,"NAME : %s",nameOfDataset);
			//printf("Name of dataset is: %s\n", nameOfDataset);
			got_name = 1;
		}
		else
		{
			//printf("\n Name not found. Invalid dataset. Exiting code. \n");
			fclose(vrp_input_file);
			return 1;
		}
	}

	// Comment

	if(fgets(lineInFile, 1024, vrp_input_file))
	{
		if(strcmp(lineInFile, "COMMENT")!= 0)
		{
			//printf("\nReceived Line: %s", lineInFile);
			sscanf(lineInFile,"COMMENT : %[^\t\n]",commentOfDataset);
			//printf("COMMENT of dataset is: %s\n", commentOfDataset);
		}
		else
		{
			//printf("\n Comment not found. Invalid dataset. Exiting code. \n");
			fclose(vrp_input_file);
			return 1;
		}
	}

	// Type

	if(fgets(lineInFile, 1024, vrp_input_file))
	{
		if(strcmp(lineInFile, "TYPE")!= 0)
		{
			//printf("\nReceived Line: %s", lineInFile);
			sscanf(lineInFile,"TYPE : %s",typeOfDataset);
			//printf("Type of dataset is: %s\n", typeOfDataset);
		}
		else
		{
			//printf("\n Type not found. Invalid dataset. Exiting code. \n");
			fclose(vrp_input_file);
			return 1;
		}
	}

	// Dimension

	if(fgets(lineInFile, 1024, vrp_input_file))
	{
		if((strcmp(lineInFile, "DIMENSION")!= 0) && !(got_dimension))
		{
			//printf("\nReceived Line: %s", lineInFile);
			sscanf(lineInFile,"DIMENSION : %d",&totalNodes);
			//printf("Nodes of dataset is: %d\n", totalNodes);
			got_dimension = 1;
		}
		else
		{
			//printf("\n Dimension not found. Invalid dataset. Exiting code. \n");
			fclose(vrp_input_file);
			return 1;
		}
	}

	// Edge

	if(fgets(lineInFile, 1024, vrp_input_file))
	{
		if(strcmp(lineInFile, "EDGE_WEIGHT_TYPE")!= 0)
		{
			//printf("\nReceived Line: %s", lineInFile);
			sscanf(lineInFile,"EDGE_WEIGHT_TYPE : %s",edgeTypeOfDataset);
			//printf("Edge of dataset is: %s\n", edgeTypeOfDataset);
		}
		else
		{
			//printf("\n Edge not found. Invalid dataset. Exiting code. \n");
			fclose(vrp_input_file);
			return 1;
		}
	}

	// Capacity

	if(fgets(lineInFile, 1024, vrp_input_file))
	{
		if((strcmp(lineInFile, "CAPACITY")!= 0) && !(got_capacity))
		{
			//printf("\nReceived Line: %s", lineInFile);
			sscanf(lineInFile,"CAPACITY : %d", &capacitOfVehicle);
			//printf("Capacity of dataset is: %d\n", capacitOfVehicle);
			got_capacity = 1;
		}
		else
		{
			//printf("\n Capacity not found. Invalid dataset. Exiting code. \n");
			fclose(vrp_input_file);
			return 1;
		}
	}

	// Co-ordinates

	if(fgets(lineInFile, 1024, vrp_input_file))
	{
		if((strcmp(lineInFile, "NODE_COORD_SECTION")!= 0) && !(got_co_ord))
		{
			//printf("\nReceived Line: %s\n", lineInFile);
			got_co_ord = 1;
		}
		else
		{
			//printf("\n COORD data not found. Invalid dataset. Exiting code. \n");
			fclose(vrp_input_file);
			return 1;
		}
	}

	size_t nodeArraySize = (totalNodes + 1) * sizeof(nodeInfo);

	struct nodeInfo *vrpNodeInfo_h = (nodeInfo *) malloc (nodeArraySize);

	vrpNodeInfo_h[0].node	= 0;
	vrpNodeInfo_h[0].xCoOrd	= 0;
	vrpNodeInfo_h[0].yCoOrd	= 0;
	vrpNodeInfo_h[0].demand	= 0;

	for(int i = 1; i <= totalNodes; i++)
	{
		if(fgets(lineInFile, 1024, vrp_input_file))
		{
			if(isNotAlpha(lineInFile) && got_co_ord)
			{
				//printf("Received Line: %s", lineInFile);
				sscanf(lineInFile,"%d %d %d",&vrpNodeInfo_h[i].node,&vrpNodeInfo_h[i].xCoOrd,&vrpNodeInfo_h[i].yCoOrd);
				//printf("Node : %2d, xCoOrd: %2d, yCoOrd: %2d\n\n", vrpNodeInfo_h[i].node, vrpNodeInfo_h[i].xCoOrd,vrpNodeInfo_h[i].yCoOrd);
			}
			else
			{
				//printf("\n COORD data Incomplete. Invalid dataset. Exiting code. \n");
				fclose(vrp_input_file);
				return 1;
			}
		}
	}

	// Demand data

	if(fgets(lineInFile, 1024, vrp_input_file))
	{
		if((strcmp(lineInFile, "DEMAND_SECTION")!= 0) && !(got_demand_val))
		{
			//printf("Received Line: %s\n", lineInFile);
			got_demand_val = 1;
		}
		else
		{
			//printf("\n COORD data not found. Invalid dataset. Exiting code. \n");
			fclose(vrp_input_file);
			return 1;
		}
	}

	for(int i = 1; i <= totalNodes; i++)
	{
		if(fgets(lineInFile, 1024, vrp_input_file))
		{
			if(isNotAlpha(lineInFile) && got_co_ord)
			{
				int nodeval;
				int demandval;
				//printf("Received Line: %s", lineInFile);
				sscanf(lineInFile,"%d %d", &nodeval, &demandval);
				if(vrpNodeInfo_h[i].node == nodeval)
				{
					vrpNodeInfo_h[i].demand = demandval;
					//printf("Node : %2d, xCoOrd: %2d, yCoOrd: %2d, Demand: %3d\n\n", vrpNodeInfo_h[i].node, vrpNodeInfo_h[i].xCoOrd,vrpNodeInfo_h[i].yCoOrd,vrpNodeInfo_h[i].demand);
				}
				else
				{
					//printf("\n COORD data not in sequence. Invalid dataset. Exiting code. \n");
					fclose(vrp_input_file);
					return 1;
				}
			}
			else
			{
				//printf("\n COORD data Incomplete. Invalid dataset. Exiting code. \n");
				fclose(vrp_input_file);
				return 1;
			}
		}
	}
	fclose(vrp_input_file);
	printf("Dataset Loading Completed..!!\n");

	clock_t startTime = clock();

	int rows = totalNodes+1, columns = totalNodes+1;
	
	// Print the vector length to be used, and compute its size
	size_t size = rows * columns * sizeof(int);
	
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Initialize the host costMatrix_h

	int *costMatrix_h = (int *) malloc(size);
	
	// Allocate the device input vector A
	int *costMatrix_d = (int *) malloc(size);
	
	//(void **)devptr pointer to allocated device memory
	err = cudaMalloc((void **)&costMatrix_d, size);
	////printf("Allocated costMatrix_d memory\n");
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	struct nodeInfo *vrpNodeInfo_d = (struct nodeInfo *)malloc(nodeArraySize);

	err = cudaMalloc((void **)&vrpNodeInfo_d, nodeArraySize);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	////printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(vrpNodeInfo_d, vrpNodeInfo_h, nodeArraySize, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	dim3 threadsPerBlock(MaxThreadx, MaxThready);
	dim3 blocksPerGrid((((columns + MaxThreadsPerBlock - 1)/MaxThreadsPerBlock) + 1), (((rows + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock) + 1));
	//printf("CUDA kernel launch with %d, %d blocks \n", blocksPerGrid.x, blocksPerGrid.y);

	getCostMatrix <<< blocksPerGrid, threadsPerBlock >>>((struct nodeInfo *)vrpNodeInfo_d, (int *)costMatrix_d, rows, columns);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch savingsCalc kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	////printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(costMatrix_h, costMatrix_d, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//printf("\nPrinting the costMatrix_h below:\n\n");
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < columns; ++j) {
			//printf("%2d\t", *(costMatrix_h + i*columns + j));
		}
		//printf("\n");
	}

	int *savingsMatrix_h = (int*) malloc(rows * columns * sizeof(int));

	// Allocate the device output vector C
	int *savingsMatrix_d = (int *)malloc(rows * columns * sizeof(int));
	err = cudaMalloc((void **)&savingsMatrix_d, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&costMatrix_d, size);
	////printf("Allocated costMatrix_d memory\n");
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	////printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(costMatrix_d, costMatrix_h, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	//( void* dst, const void* src, size_t count, cudaMemcpyKind kind )

	savingsCalc <<< blocksPerGrid, threadsPerBlock >>>((int *)costMatrix_d,(int *) savingsMatrix_d, rows,columns);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch savingsCalc kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	////printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(savingsMatrix_h, savingsMatrix_d, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	struct savings_info *records_h = (savings_info*)malloc(rows * columns * sizeof(savings_info));

	int count = 0;
	// Verify that the result vector is correct
	//printf("\nPrinting the savingsMatrix_h below:\n\n");
	for (int i = 1; i < rows-1; ++i)
	{
		for (int j = 2; j < columns; ++j) {
			if (i < j) {
			//printf("%2d\t", *(savingsMatrix_h + i*columns + j));
			records_h[count].startNode = i;
			records_h[count].endNode = j;
			records_h[count].savingsBetweenNodes = *(savingsMatrix_h + i*columns + j);
			count++;
			}

		}
		//printf("\n");
	}
	//printf("\n");

	//Parallel Sorting
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
	dim3 threadsPerBlock_savingsSort(MaxThreadsPerBlock);
	int blocksPerGrid_savingsSort = (count + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;;//((numNodes*numNodes) + threadsPerBlock - 1) / threadsPerBlock;
	////printf("RADHIKA : CUDA kernel launch with %d blocks \n", blocksPerGrid);

	//savingsCalc << < blocksPerGrid, threadsPerBlock >> >((int *)costMatrix_d, (int *)savingsMatrix_d, rows, columns);
	sortSavings <<< blocksPerGrid_savingsSort, threadsPerBlock_savingsSort >>>((savings_info *)records_d, count);
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
		//printf("Sorted Record %2d = {%2d\t, %2d\t,%2d\t} ", i, records_h[i].startNode, records_h[i].endNode, records_h[i].savingsBetweenNodes);
		//printf("\n");
	}

	//****************************************************************************//
	/*************MAIN AALGORITHM : CLARKE AND WRITE ******************************/
	//****************************************************************************//

	// Result dictionary to capture node visit updates
	int vehicleCapacity = capacitOfVehicle;
	int node_count = totalNodes + 1;
	int maxRouteCount = totalNodes;
	struct keyVal *resultDict_h = (keyVal *)malloc(node_count * sizeof(keyVal));

	for (int i = 0; i < node_count; i++)
	{
		resultDict_h[i].key = vrpNodeInfo_h[i].node;
		resultDict_h[i].val = 0;
	}

	// Result list to store final paths
	struct route *resultList_h = (route *)malloc(maxRouteCount * sizeof(route));

	int nodesProcessed = 0;
	int routesAdded = 0;

	////printf("\nNo of elements of  records_h :%d \n\n" , count);

	for (int z = 0; z < count; z++) {
		int edge_i = records_h[z].startNode;
		int edge_j = records_h[z].endNode;

		int demandParam_h_edge_i_dem;
		int demandParam_h_edge_j_dem;

		demandParam_h_edge_i_dem = vrpNodeInfo_h[edge_i].demand;
		//printf("\n\nDemand of %d is %d\n", edge_i, demandParam_h_edge_i_dem);

		demandParam_h_edge_j_dem = vrpNodeInfo_h[edge_j].demand;
		//printf("\nDemand of %d is %d\n", edge_j, demandParam_h_edge_j_dem);

		//printf("\nDemands : %d and %d for Edges : %d and %d \n", demandParam_h_edge_i_dem, demandParam_h_edge_j_dem, edge_i, edge_j);

		if (nodesProcessed != 0) {
			if (demandParam_h_edge_i_dem + demandParam_h_edge_j_dem <= vehicleCapacity) {
				//printf("Iteration No.: %d for %d , %d\n", z, edge_i, edge_j);

				if (resultDict_h[edge_i].val == 1 && resultDict_h[edge_j].val == 0)
				{
					int indexOfRoute = resultDict_h[edge_i].routeIndex;
					int numberOfNodesInRoute = resultList_h[indexOfRoute].nodesAdded;
					int total_demand = 0;
					total_demand += demandParam_h_edge_j_dem;
					for (int temp_i = 0; temp_i < numberOfNodesInRoute; temp_i++)
					{
						total_demand += vrpNodeInfo_h[resultList_h[indexOfRoute].nodes_in_route[temp_i]].demand;
					}
					//printf("Total Demand: %d\n", total_demand);
					if (total_demand <= vehicleCapacity)
					{
						if (resultDict_h[edge_i].indexOfnodeInRouteInResultArray == 0 || resultDict_h[edge_i].indexOfnodeInRouteInResultArray == (resultList_h[indexOfRoute].nodesAdded - 1))
						{
							resultList_h[indexOfRoute].nodes_in_route[numberOfNodesInRoute] = edge_j;
							resultList_h[indexOfRoute].nodesAdded += 1;
							resultDict_h[edge_j].val = 1;
							resultDict_h[edge_j].routeIndex = indexOfRoute;
							resultDict_h[edge_j].indexOfnodeInRouteInResultArray = numberOfNodesInRoute;
							nodesProcessed += 1;
							//printf("i. Added Node %d after %d\n", edge_j, edge_i);
						}
						else
						{
							//printf("i. Can't add the %d node as %d is intermediate node \n", edge_j, edge_i);
						}
					}
					else
					{
						//printf("i. Capacity exceeding for nodes %d and %d \n" , edge_i, edge_j);
					}
				}
				else if (resultDict_h[edge_i].val == 0 && resultDict_h[edge_j].val == 1)
				{
					int indexOfRoute = resultDict_h[edge_j].routeIndex;
					int numberOfNodesInRoute = resultList_h[indexOfRoute].nodesAdded;
					int total_demand = 0;
					total_demand += demandParam_h_edge_i_dem;
					for (int temp_i = 0; temp_i < numberOfNodesInRoute; temp_i++)
					{
						total_demand += vrpNodeInfo_h[resultList_h[indexOfRoute].nodes_in_route[temp_i]].demand;
					}
					//printf("Total Demand: %d\n", total_demand);
					if (total_demand <= vehicleCapacity)
					{
						if (resultDict_h[edge_j].indexOfnodeInRouteInResultArray == 0 || resultDict_h[edge_j].indexOfnodeInRouteInResultArray == (resultList_h[indexOfRoute].nodesAdded - 1))
						{
							resultList_h[indexOfRoute].nodes_in_route[numberOfNodesInRoute] = edge_i;
							resultList_h[indexOfRoute].nodesAdded += 1;
							resultDict_h[edge_i].val = 1;
							resultDict_h[edge_i].routeIndex = indexOfRoute;
							resultDict_h[edge_i].indexOfnodeInRouteInResultArray = numberOfNodesInRoute;
							nodesProcessed += 1;
							//printf("j. Added Node %d after %d\n", edge_i, edge_j);
						}
						else
						{
							//printf("j.Can't add the %d node as %d is intermediate node \n", edge_i, edge_j);
						}
					}
					else
					{
						//printf("j. Capacity exceeding for nodes %d and %d \n", edge_i, edge_j);
					}
				}
				else if (resultDict_h[edge_i].val == 0 && resultDict_h[edge_j].val == 0)
				{
					resultList_h[routesAdded].nodes_in_route[0] = edge_i;
					resultList_h[routesAdded].nodes_in_route[1] = edge_j;
					resultList_h[routesAdded].nodesAdded = 2;
					resultDict_h[edge_i].val = 1;
					resultDict_h[edge_j].val = 1;
					resultDict_h[edge_i].routeIndex = routesAdded;
					resultDict_h[edge_j].routeIndex = routesAdded;
					resultDict_h[edge_i].indexOfnodeInRouteInResultArray = 0;
					resultDict_h[edge_j].indexOfnodeInRouteInResultArray = 1;
					nodesProcessed += 2;
					routesAdded += 1;
					//printf("Added both nodes %d and %d\n", edge_i, edge_j);
					//printf("Route Added : numberOfRoutes = %d \n", routesAdded);
				}
				else
				{
					//printf("Nodes %d and %d are already processed\n", edge_i, edge_j);
				}
			}
			else
			{
				//printf("Iteration No.: %d for %d , %d \n", z, edge_i, edge_j);
				//printf("Capacity Constraints Violated : %d \n ", demandParam_h_edge_i_dem + demandParam_h_edge_j_dem);
			}
		}
		else
		{
			if (demandParam_h_edge_i_dem + demandParam_h_edge_j_dem <= vehicleCapacity)
			{
				//printf("Iteration No.: %d for %d , %d \n", z, edge_i, edge_j);
				//printf("2. Capacity Constraints Not Violated : %d \n", demandParam_h_edge_i_dem + demandParam_h_edge_j_dem);

				resultList_h[routesAdded].nodes_in_route[0]  = edge_i;
				resultList_h[routesAdded].nodes_in_route[1]  = edge_j;
				resultList_h[routesAdded].nodesAdded = 2;
				resultDict_h[edge_i].val = 1;
				resultDict_h[edge_j].val = 1;
				resultDict_h[edge_i].routeIndex = routesAdded;
				resultDict_h[edge_j].routeIndex = routesAdded;
				resultDict_h[edge_i].indexOfnodeInRouteInResultArray = 0;
				resultDict_h[edge_j].indexOfnodeInRouteInResultArray = 1;
				nodesProcessed += 2;
				routesAdded += 1;
				//printf(" Route Added numberOfRoutes = %d %d,%d\n ", routesAdded, edge_i, edge_j);
			}
		}
	}//End of for_count

	for (int i = 1; i < node_count; i++)
	{
		if (resultDict_h[i].val == 0)
		{
			resultList_h[routesAdded].nodes_in_route[0] = vrpNodeInfo_h[i].node;
			resultList_h[routesAdded].nodesAdded = 1;
			nodesProcessed += 1;
			routesAdded += 1;
		}

	}

	clock_t endTime = clock();
	int timeSpent = (int)(((endTime - startTime) * 1000) / CLOCKS_PER_SEC);
	
	FILE* routes_file = fopen("routes.txt", "w");
	int totalSavings = 0;

	printf("\n************* Final Result *****************\n ");
	
	printf("\nTime required for computing is %d microseceonds\n", timeSpent);
	fprintf(routes_file, "Name of Dataset: %s\n", nameOfDataset);
	fprintf(routes_file, "Time for compute : %d us\n", timeSpent);
	for (int z = 0; z < routesAdded; z++)
	{
		struct route temproute = resultList_h[z];
		int localSavings = 0;
		int node1 = 0;
		int node2 = 0;
		int decisionMaker = 0;
		fprintf(routes_file, "Route %d : ", z);
		printf("\nRoute\t\t: %d\n", z);
		printf("NodesAdded\t: %d\n\n[\t", temproute.nodesAdded);

		 for (int i = 0; i < temproute.nodesAdded; i++)
		 {
			 	 printf("%d \t", temproute.nodes_in_route[i]);

				 if (i == temproute.nodesAdded - 1) {
					 fprintf(routes_file, "%d", temproute.nodes_in_route[i]);
				 }
				 else
				 {
					 fprintf(routes_file, "%d, ", temproute.nodes_in_route[i]);
				 }
				 

				 if (decisionMaker == 0)
				 {
					 if (node1 != 0) 
					 {
						 node1 = temproute.nodes_in_route[i];
						 localSavings += *(savingsMatrix_h + node2*columns + node1);
					 }
					 else
					 {
						 node1 = temproute.nodes_in_route[i];
					 }
					 decisionMaker = 1;
				 }
				 else
				 {
					 node2 = temproute.nodes_in_route[i];
					 decisionMaker = 0;
					 localSavings += *(savingsMatrix_h + node1*columns + node2);
				 }
		 }
		 if (node2 == 0)
		 {
			 localSavings = *(savingsMatrix_h + node1);
		 }
		 printf("]\n");
		 decisionMaker = 0;
		 totalSavings += localSavings;
		 printf("Savings: %d\n", localSavings);
		 fprintf(routes_file,"\nSavings: %d\n", localSavings);
	}

	printf("\nTotal Nodes Processed: %d\n", nodesProcessed);
	fprintf(routes_file, "Total Nodes Processed: %d\n", nodesProcessed);
	printf("\nTotal Savings: %d\n", totalSavings);
	fprintf(routes_file, "Total Savings: %d", totalSavings);
	fclose(routes_file);
	printf("\n********************************************\n ");

	//****************************************************************************//
	//***********************End of Main Algorithm********************************//
	//****************************************************************************//

	// Free device global memory
	err = cudaFree(vrpNodeInfo_d);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

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

	err = cudaFree(records_d);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(costMatrix_h);
	free(vrpNodeInfo_h);
	free(resultDict_h);
	free(resultList_h);
	free(savingsMatrix_h);
	free(records_h);

	//scanf("");
	getchar();
	return 0;
}
