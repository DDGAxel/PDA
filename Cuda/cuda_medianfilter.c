#include "cuda_runtime.h"
#include <stdio.h>
#include <memory.h>

#define N 33 * 1024
#define threadsPerBlock 256
#define blocksPerGrid (N + threadsPerBlock - 1) / threadsPerBlock
#define RADIUS 2

typedef int element;

__global__ void _medianfilter(const element* signal, element* result)
{
	__shared__ element cache[threadsPerBlock + 2 * RADIUS];
	element window[5];
	int gindex = threadIdx.x + blockDim.x * blockIdx.x;
	int lindex = threadIdx.x + RADIUS;
	cache[lindex] = signal[gindex];
	if (threadIdx.x < RADIUS)
	{
		cache[lindex - RADIUS] = signal[gindex - RADIUS];
		cache[lindex + threadsPerBlock] = signal[gindex + threadsPerBlock];
	}
	__syncthreads();
	for (int j = 0; j < 2 * RADIUS + 1; ++j)
		window[j] = cache[threadIdx.x + j];

	for (int j = 0; j < RADIUS + 1; ++j)
	{
		int min = j;
		for (int k = j + 1; k < 2 * RADIUS + 1; ++k)
			if (window[k] < window[min])
				min = k;
		const element temp = window[j];
		window[j] = window[min];
		window[min] = temp;
	}
	result[gindex] = window[RADIUS];
}

void medianfilter(element* signal, element* result)
{
	element *dev_extension, *dev_result;

	if (!signal || N < 1)
		return;
	if (N == 1)
	{
		if (result)
			result[0] = signal[0];
		return;
	}

	element* extension = (element*)malloc((N + 2 * RADIUS) * sizeof(element));

	if (!extension)
		return;

	cudaMemcpy(extension + 2, signal, N * sizeof(element), cudaMemcpyHostToHost);
	for (int i = 0; i < RADIUS; ++i)
	{
		extension[i] = signal[1 - i];
		extension[N + RADIUS + i] = signal[N - 1 - i];
	}

	cudaMalloc((void**)&dev_extension, (N + 2 * RADIUS) * sizeof(int));
	cudaMalloc((void**)&dev_result, N * sizeof(int));

	cudaMemcpy(dev_extension, extension, (N + 2 * RADIUS) * sizeof(element), cudaMemcpyHostToDevice);
	
	for (int i = 0; i < 10; ++i)
		_medianfilter<<<blocksPerGrid, threadsPerBlock>>>(dev_extension + RADIUS, dev_result);

	cudaMemcpy(result, dev_result, N * sizeof(element), cudaMemcpyDeviceToHost);

	free(extension);
	cudaFree(dev_extension);
	cudaFree(dev_result);
}

int main()
{
	int *Signal, *result;
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	FILE *fp;
	
	Signal = (int *)malloc(N * sizeof(int));
	result = (element *)malloc(N * sizeof(element));
	
	for (int i = 0; i < N; i++)
	{
		Signal[i] = i % 5 + 1;
	}
	cudaEventRecord(start, 0);
	medianfilter(Signal, result);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%.3lf ms\n", elapsedTime);

	fp = fopen("result.txt", "w");
	if (fp == NULL)
		printf("OPEN FILE FAILS!\n");
	for (int i = 0; i < N; i ++)
		fprintf(fp, "%d ", result[i]);

	fclose(fp);
	return 0;
}