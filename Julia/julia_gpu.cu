ya/*
References:
[1] http://stackoverflow.com/questions/23711681/generating-custom-color-palette-for-julia-set
[2] http://www.cs.rit.edu/~ncs/color/t_convert.html

Another source I referred to:
[3] Book: CUDA By Example: http://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf
	The cuComplex data structure being used in this code has been taken from the above source, with slight modifications.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <string.h>
#include <IL/il.h>
#include <IL/ilu.h>
#include <time.h>
//#include <Windows.h> //For GetTickCount() call to get time for benchmarking purposes 

using namespace std;

#define N 4096
#define SQRT_2 1.4142
#define MAX_ITER 512

struct cuComplex {
	float r;
	float i;
	__host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}
	__host__ __device__ float magnitude2(void) {
		return r * r + i * i;
	}
	__host__ __device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__host__ __device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

void saveImage(int width, int height, unsigned char * bitmap, cuComplex seed);
__global__ void compute_julia_gpu(unsigned char* image);
__device__ void HSVtoRGB_GPU(float *r, float *g, float *b, float h, float s, float v);
__device__ int julia_set(int x, int y);

int main(int argc, char **argv)
{
	cuComplex c(0.285f, 0.01f);
	//cuComplex c(-0.8f, 0.156f); Another interesting value of c.
	if (argc > 2)
	{
		c.r = atof(argv[1]);
		c.i = atof(argv[2]);
	}
	else
	{
		fprintf(stderr, "Usage: %s <real> <imag>\nWhere <real> and <imag> form the complex seed for the Julia set.\n", argv[0]);
	}
	ilInit();

	dim3 grid(N, N); //Creating a grid of NxN dimensions
	unsigned char *image = new unsigned char[N*N * 3]; //RGB image for the CPU (host)
	size_t size = sizeof(unsigned char) * N * N * 3;
	unsigned char *d_image; //RGB image for the GPU (device)

	cudaError_t cudaStatus = cudaMalloc((void **)&d_image, size);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc for GPU failed!");
	}

	//DWORD getTime = GetTickCount(); //Timing data. Uses Windows.h, will not work on Linux
	compute_julia_gpu <<<grid, 128>>>(d_image); //2D grid with 1D blocks, with 128 threads each
	cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);
	//printf("The time taken is: %d ms\n", GetTickCount() - getTime);
	saveImage(N, N, image, c);
	cudaFree(d_image);	//Freeing up the memory
	delete[] image;
}

__device__ int julia_set(int x, int y) //Helper function to check if (x, y) belong to Julia set. 
{
	cuComplex c(0.285f, 0.01f);
	//cuComplex c(-0.8f, 0.156f);
	cuComplex z_old(0.0, 0.0);
	cuComplex z_new(0.0, 0.0);
	int ret_val = MAX_ITER;
	z_new.r = (4.0f * x / (N)-2.0f);
	z_new.i = (4.0f * y / (N)-2.0f);
	for (int i = 0; i < MAX_ITER; i++)
	{
		z_old.r = z_new.r;
		z_old.i = z_new.i;
		z_new = (z_new * z_new) + c;
		if (z_new.magnitude2() > 4.0f)
		{
			ret_val = i;
			break;
		}
	}
	return ret_val;
}
__global__ void compute_julia_gpu(unsigned char* image) 
{
	// Map x and y accordingly
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	// Check if allotted x and y values aren't out of bound

	if (x < N && y < N)
	{
		int i = julia_set(x, y);
		float brightness = (i<MAX_ITER) ? 1.0f : 0.0f;
		float hue = (i % MAX_ITER) / float(MAX_ITER - 1);
		hue = (120 * sqrtf(hue) + 150);
		float r, g, b;
		HSVtoRGB_GPU(&r, &g, &b, hue, 1.0f, brightness);
		image[(x + y*N) * 3 + 0] = (unsigned char)(b * 255);
		image[(x + y*N) * 3 + 1] = (unsigned char)(g * 255);
		image[(x + y*N) * 3 + 2] = (unsigned char)(r * 255);
	}
}

void saveImage(int width, int height, unsigned char * bitmap, cuComplex seed)
{
	ILuint imageID = ilGenImage();
	ilBindImage(imageID);
	ilTexImage(width, height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, bitmap);
	//ilEnable(IL_FILE_OVERWRITE);
	char imageName[256];
	sprintf(imageName, "Julia %.3f + i%.3f.png", seed.r, seed.i);
	ilSave(IL_PNG, imageName);
	fprintf(stderr, "Image saved as: %s\n", imageName);
}

__device__ void HSVtoRGB_GPU(float *r, float *g, float *b, float h, float s, float v)
{
	int i;
	float f, p, q, t;
	if (s == 0) {
		// achromatic (grey)
		*r = *g = *b = v;
		return;
	}
	h /= 60;			// sector 0 to 5
	i = floor(h);
	f = h - i;			// factorial part of h
	p = v * (1 - s);
	q = v * (1 - s * f);
	t = v * (1 - s * (1 - f));
	switch (i) {
	case 0:
		*r = v;
		*g = t;
		*b = p;
		break;
	case 1:
		*r = q;
		*g = v;
		*b = p;
		break;
	case 2:
		*r = p;
		*g = v;
		*b = t;
		break;
	case 3:
		*r = p;
		*g = q;
		*b = v;
		break;
	case 4:
		*r = t;
		*g = p;
		*b = v;
		break;
	default:		// case 5:
		*r = v;
		*g = p;
		*b = q;
		break;
	}
}
