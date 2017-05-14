#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#define MAXIMUM_VALUE 1000000.0
#define HANDLE_ERROR( err )  ( HandleError( err, __FILE__, __LINE__ ) )
#define BLOCK_SIZE 1024


void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);

		exit(EXIT_FAILURE);
	}
}

void checkCUDAError(const char *msg, bool exitOnError) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		if (exitOnError) {
			exit(-1);
		}
	}
}

__global__ void kernel_calculate_sum(double * dev_array_sums,
	unsigned int array_size,
	double * dev_block_sums) {
		
	__shared__ double shared_sum[BLOCK_SIZE];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < array_size)
	{
		shared_sum[tid] = dev_array_sums[i];
	}
	else
	{
		shared_sum[tid] = 0;
	}
	__syncthreads();
	
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid < s) {
			shared_sum[tid] += shared_sum[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0)
	{
		dev_block_sums[blockIdx.x] = shared_sum[0];
	}
}

int main(int argc, char* argv[]) {

	unsigned int array_size, seed, i;
	float runtime;

	if (argc < 3) {
		printf("Format: reduction_large_array <size of array> <random seed>\n");
		printf("Arguments:\n");
		printf("  size of array - This is the size of the array to be generated and processed\n");
		printf("  random seed   - This integer will be used to seed the random number\n");
		printf("                  generator that will generate the contents of the array\n");
		printf("                  to be processed\n");

		exit(1);
	}
	array_size = atoi(argv[1]);
	seed = atoi(argv[2]);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	double *array = (double *)malloc(array_size * sizeof(double));

	srand(seed);
	for (i = 0; i < array_size; i++)
		array[i] = ((double)rand() / (double)RAND_MAX) * MAXIMUM_VALUE;
	
	double sum = 0;

	unsigned int num_threads = BLOCK_SIZE;
	unsigned int num_blocks = (array_size + num_threads - 1) / num_threads;

	double * dev_array;
	HANDLE_ERROR(cudaMalloc((void**)& dev_array, array_size * sizeof(double)));

	HANDLE_ERROR(cudaMemcpy(dev_array, array, array_size * sizeof(double), cudaMemcpyHostToDevice));

	unsigned int loop_num = 0;
	unsigned int current_size = array_size;
	do
	{
		current_size = (current_size + num_threads - 1) / num_threads;
		loop_num++;
	} while (current_size > 1);

	double ** dev_block_sums;

	dev_block_sums = (double**)malloc((loop_num + 1) * sizeof(double*));

	current_size = array_size;
	for (unsigned int i = 0; i < loop_num; i++)
	{
		current_size = (current_size + num_threads - 1) / num_threads;
		cudaMalloc((void**)& dev_block_sums[i], current_size * sizeof(double));
	}

	double * dev_array_sums_in = dev_array;

	double * dev_block_sums_out = dev_block_sums[0];

	dev_block_sums_out = dev_block_sums[0];
	current_size = array_size;
	for (unsigned int i = 0; i < loop_num; i++)
	{
		kernel_calculate_sum << <num_blocks, num_threads >> >(dev_array_sums_in,
			current_size,
			dev_block_sums_out);

		checkCUDAError("kernel_calculate_sum", true);
		current_size = (current_size + num_threads - 1) / num_threads;
		num_blocks = (current_size + num_threads - 1) / num_threads;
		dev_array_sums_in = dev_block_sums_out;
		dev_block_sums_out = dev_block_sums[i + 1];
	}

	cudaMemcpy(&sum, dev_array_sums_in, sizeof(double), cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < loop_num; i++)
	{
		cudaFree(dev_block_sums[i]);
	}

	free(dev_block_sums);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&runtime, start, stop);

	printf("Statistics for array ( %d, %d ):\n", array_size, seed);
	printf("Sum result: %f\n", sum);
	printf("Processing Time: %4.4f milliseconds\n", runtime);

	free(array);

	return 0;
}