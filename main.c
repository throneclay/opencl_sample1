#include <stdio.h>
#include <stdlib.h>
#include <mm_malloc.h>
#include <time.h>
#include "OCLKernel.h"

int generateImageFromFile(float*d, int Wl, int Hl, const char* fileName)
{
    FILE *fin=fopen(fileName, "r");
    for(int i=0; i<Wl*Hl; ++i) {
        fscanf(fin, "%f", &d[i]);
    }
    fclose(fin);
}

int main(int argc, char const *argv[])
{
	float *images, *result, *filter;

	int ITER=1;
	int w,h;
	w = (WIDTH-8)/WSTEP+1;
	h = (HEIGHT-4)/HSTEP+1;

	images = (float*)_mm_malloc(sizeof(float) * WIDTH * HEIGHT * CHANNEL * 2, 64);
	result = (float*)_mm_malloc(sizeof(float) * w * h * 64 * 2 * FILTERNUM, 64);
	filter = (float*)_mm_malloc(sizeof(float) * 64 * FILTERNUM * CHANNEL ,64);
	for (int i = 0; i < WIDTH * HEIGHT * CHANNEL; ++i)
	{
		images[i] = i;
	}
	for(int i = 0; i < 32 * FILTERNUM * CHANNEL; ++i)
	{
		filter[i] = 1.0;
	}

	printf("w = %d h = %d h/4 = %d\n", w,h,h/4);
	double dtime = callKernel(images, filter, result, w,h,ITER, "Kernel.cl","conv4x8");
	printf("time is %lfms\n",1000*dtime);

	printf("Speed achieve %0.2lf GFLOPS\n", 
		(1e-9*WIDTH*HEIGHT*9*CHANNEL*FILTERNUM*4*ITER)/dtime);
	
	_mm_free(images);
	_mm_free(filter);
	_mm_free(result);
	return 0;
}
