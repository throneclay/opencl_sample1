#ifndef __OCLKERNEL_H__
#define __OCLKERNEL_H__

#define FILTERNUM 32
double callKernel(float*images, float* filter, float*result, 
	int w,int h,int ITER,
 	const char* KernelName, const char* KernelFunc);

#endif
