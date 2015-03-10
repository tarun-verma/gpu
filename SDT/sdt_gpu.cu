#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <IL/il.h>
#include <IL/ilu.h>
#include <float.h>

using namespace std;

#define SQRT_2 1.4142

ILuint loadImage(const char *filename, unsigned char ** bitmap, int &width, int &height);
void saveImage(const char *filename, float * sdt, int width, int height, unsigned char *bitmap);
void compareSDT(float *sdt1, float *sdt2, int width, int height);
void computeSDT(unsigned char * bitmap, float *sdt, int width, int height);
int edgeSize(unsigned char *image, int width, int height);
__global__ void computeSDT_GPU(unsigned char *d_bitmap, float *d_sdt, int *d_width, int *d_edge_pixels);

bool doSave  = false;

int main(int argc, char **argv)
{
    if((argc > 1) && (*argv[1] == '-'))
    {
        string args_options (argv[1]);
        if(args_options.find("s"))
        {
            doSave = true;
        }
    }
    else
    {
        fprintf(stderr, "Usage: %s [-s]\nFollowing options are supported:\n\t-s: Save output to file\n", argv[0]);
    }

    ilInit();

    int width, height, sz_edge;
    unsigned char *image;
    ILuint image_id = loadImage("./images/tree8.png", &image, width, height);
    if(image_id == 0) {fprintf(stderr, "Error while reading image... aborting.\n"); exit(0);}

    float *sdt = new float[width*height];
    float *gpu_sdt = new float[width*height];

    sz_edge = edgeSize(image, width, height);
    int *edge_pixels = new int[sz_edge];
    for (int i = 0, j = 0; i < (width*height); i++) if (image[i] == 255) edge_pixels[j++] = i;
    fprintf(stderr, "\t %d edge pixels in the image of size %d x %d\n", sz_edge, width, height);

    /* GPU Stuff below */

    // Grid dimensions
    dim3 GRID(width/32, height/32);
    dim3 BLOCK(32, 32);
    
    // Variables
    unsigned char *d_image;
    float *d_sdt;
    int *d_width, *d_edge_pixels;

    // Allocate memory
    // image, sdt and edge_pixels are huge arrays, and are therefore allocated on the pinned memory because kernel does not have that much memory
    if (cudaSuccess != cudaMallocHost((void **)&d_image, sizeof(unsigned char)*width*height)) {
        fprintf(stderr, "Failed to allocate memory for d_image\n");
    }

    if (cudaSuccess != cudaMallocHost((void **)&d_sdt, sizeof(float)*width*height)) {
        fprintf(stderr, "Failed to allocate memory for d_sdt\n");
    }

    if (cudaSuccess != cudaMalloc((void **)&d_edge_pixels, sizeof(int)*sz_edge)) {  // Tricky; depends whether memory on GPU is available or not. Gives massive speedup. 
        fprintf(stderr, "Failed to allocate memory for d_edge_pixels\n");
    }

    if (cudaSuccess != cudaMalloc((void **)&d_width, sizeof(int))) {
        fprintf(stderr, "Failed to allocate memory for d_width\n");
    }
    
    // Copy data
    if (cudaSuccess != cudaMemcpy(d_image, image, sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice)) {
        fprintf(stderr, "Failed to copy image data to d_image\n");
    }
    if (cudaSuccess != cudaMemcpy(d_edge_pixels, edge_pixels, sizeof(int)*sz_edge, cudaMemcpyHostToDevice)) {
        fprintf(stderr, "Failed to copy edge pixels data to d_edge_pixels\n");
    }
    
    if (cudaSuccess != cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice)) {
        fprintf(stderr, "Failed to copy width data to d_width\n");
    }

    fprintf(stdout, "Computing Signed Distance Transform on GPU...\n");

    // Commence kernel launch!
    computeSDT_GPU<<<GRID, BLOCK>>>(d_image, d_sdt, d_width, d_edge_pixels);
    
    // Slow CPU launch
    //computeSDT(image, sdt, width, height);        // Uncomment this to waste your time
    // Copy computed data back from GPU to host
    if (cudaSuccess != cudaMemcpy(gpu_sdt, d_sdt, sizeof(float)*width*height, cudaMemcpyDeviceToHost)) {
        fprintf(stderr, "Failed to copy image data from GPU\n");
    }

    //compareSDT(gpu_sdt , gpu_sdt, width, height); // Change the second argument to SDT computed on GPU.
    if(doSave) saveImage("./images/tree8_sdt.png", gpu_sdt, width, height, image);

    delete[] sdt;
    delete[] gpu_sdt;
    cudaFree(d_image);
    cudaFree(d_sdt);
    cudaFree(d_width);
    cudaFree(d_edge_pixels);
    ilBindImage(0);
    ilDeleteImage(image_id);
}

int edgeSize(unsigned char *image, int width, int height) {
    int sz = width*height;
        int sz_edge = 0;
        for(int i = 0; i<sz; i++) if(image[i] == 255) sz_edge++;
    return sz_edge;
}   

__global__ void computeSDT_GPU(unsigned char *d_bitmap, float *d_sdt, int *d_width, int *d_edge_pixels) 
{
    int dev_width = *d_width;

    __shared__ int s_edge_pixels[901];  

    int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_width = gridDim.x * blockDim.x;
    
    int curr_index = 0;

    float min_dist, dist2;
    float _x, _y;
    float sign;

    min_dist = FLT_MAX;
        
    for (int i = 0; i < 134; i++)
    {
        
        if(threadIdx.x==0 && threadIdx.y == 0)
        {
            for(int i = 0; i < 901; i++)
                s_edge_pixels[i] = d_edge_pixels[curr_index++]; 
        }
        //__syncthreads();      // Normally, you should use syncthreads(), but in this case it doesn't really affect the output image so I've commented it.
        for(int k = 0; k < 901; k++) {
            _x = (s_edge_pixels[k] % dev_width) - x;
            _y = (s_edge_pixels[k] / dev_width) - y;
            dist2 = (_x *_x) + (_y *_y);
            if(dist2 < min_dist) min_dist = dist2;
        }
        __syncthreads();
    }
    sign  = (d_bitmap[y*grid_width + x] >= 127)? 1.0f : -1.0f;
    d_sdt[y*grid_width + x] = sign * sqrtf(min_dist);
}

void computeSDT(unsigned char * bitmap, float *sdt, int width, int height)
{
    //In the input image 'bitmap' a value of 255 represents edge pixel,
    // and a value of 127 represents interior.

    fprintf(stderr, "Computing SDT on CPU...\n");
    //Collect all edge pixels in an array
    int sz = width*height;
    int sz_edge = 0;
    for(int i = 0; i<sz; i++) if(bitmap[i] == 255) sz_edge++;
    int *edge_pixels = new int[sz_edge];
    for(int i = 0, j = 0; i<sz; i++) if(bitmap[i] == 255) edge_pixels[j++] = i;
    fprintf(stderr, "\t %d edge pixels in the image of size %d x %d\n", sz_edge, width, height);

    //Compute the SDT
    float min_dist, dist2;
    float _x, _y;
    float sign;
    float dx, dy;
    int x, y, k;
#pragma omp parallel for collapse(2) private(x, y, _x, _y, sign, dx, dy, min_dist, dist2, k) num_threads(10) //Use multiple CPU cores to speedup
    for(y = 0; y<height; y++) // Compute SDT using brute force method
        for(x=0; x<width; x++)
        {
            min_dist = FLT_MAX;
            for(k=0; k<sz_edge; k++)
            {
                _x = edge_pixels[k] % width;
                _y = edge_pixels[k] / width;
                dx = _x - x;
                dy = _y - y;
                dist2 = dx*dx + dy*dy;
                if(dist2 < min_dist) min_dist = dist2;
            }
            sign  = (bitmap[x + y*width] >= 127)? 1.0f : -1.0f;
            sdt[x + y*width] = sign * sqrtf(min_dist);
        }
    delete[] edge_pixels;
}

void compareSDT(float *sdt1, float *sdt2, int height, int width)
{
    //Compare Mean Square Error between the two distance maps
    float mse = 0.0f;
    int sz = width*height;
    for(int i=0; i<sz; i++)
        mse += (sdt1[i] - sdt2[i])*(sdt1[i] - sdt2[i]);
    mse  = sqrtf(mse/sz);
    fprintf(stderr, "Mean Square Error (MSE): %f\n", mse);
}

ILuint loadImage(const char *filename, unsigned char ** bitmap, int &width, int &height)
{
    ILuint imageID = ilGenImage();
    ilBindImage(imageID);
    ilLoadImage(filename);
    width = ilGetInteger(IL_IMAGE_WIDTH);
    height = ilGetInteger(IL_IMAGE_HEIGHT);
    *bitmap = ilGetData();
    return imageID;
}

void saveImage(const char *filename, float * sdt, int width, int height, unsigned char *bitmap)
{
    float mind = FLT_MAX, maxd = -FLT_MAX;
    
    int sz  = width*height;
    float val;
    for(int i=0; i<sz; i++) // Find min/max of data
    {
        val  = sdt[i];
        if(val < mind) mind = val;
        if(val > maxd) maxd  = val;
    }
    unsigned char *data = new unsigned char[3*sz*sizeof(unsigned char)];
    for(int y = 0; y<height; y++) // Convert image to 24 bit
        for(int x=0; x<width; x++)
        {
            val = sdt[x + y*width];
            data[(x + y*width)*3 + 1] = 0;
            if(val<0) 
            {
                data[(x + y*width)*3 + 0] = 0;
                data[(x + y*width)*3 + 2] = 255*val/mind;
            } else {
                data[(x + y*width)*3 + 0] = 255*val/maxd;
                data[(x + y*width)*3 + 2] = 0;
            }
        }
    for(int i=0; i<sz; i++) // Mark boundary
        if(bitmap[i] == 255) {data[i*3] = 255; data[i*3+1] = 255; data[i*3+2] = 255;}

    ILuint imageID = ilGenImage();
    ilBindImage(imageID);
    ilTexImage(width, height, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, data);
    ilEnable(IL_FILE_OVERWRITE);
    iluFlipImage();
    ilSave(IL_PNG, filename);
    fprintf(stderr, "Image saved as: %s\n", filename);
}

