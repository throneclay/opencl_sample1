#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <sys/time.h>
#include "OCLKernel.h"

#define PRINT 0

double second()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

void MatrixT(float* d, int Wl, int Hl)
{
	float tmp;
	for(int i=0;i<Hl;i++){
		for(int j=0;j<i && j<Wl;j++){
			tmp = d[i*Wl+j];
			d[i*Wl+j] = d[j*Hl+i];
			d[j*Hl+i] = tmp;
		}
	}
}

#if PRINT
void displayMatrix(float* d, int Wl, int Hl)
{
	printf("---------------------------------------------------------------\n");
	for (int i = 0; i < Hl; ++i){
		for (int j = 0; j < Wl; ++j)
			printf("%7.1f, ",d[i*Wl+j]);
		printf("\n");
	}
}

int writeFile(float *d, int Wl, int Hl, const char* filename)
{
	FILE *writefile;
	writefile = fopen(filename, "w");
	for (int i = 0; i < Hl; ++i)
	{
		for (int j = 0; j < Wl; ++j)
		{
			fprintf(writefile,"%f\t",d[i*Wl+j]);
		}
		fprintf(writefile,"\n");
	}
	fclose(writefile);
}
#endif

typedef struct D
{
	int width;
	int height;
	int paddingW;
	int paddingH;
	cl_mem *buffer;

} imgData;

int memReshape(imgData *d,float* mem, float* image, int paddingW, int paddingH,
		cl_command_queue queue, cl_context context)
{
	cl_int err;
	printf("paddingW = %d, paddingH = %d\n", paddingW,paddingH);
	
	size_t buffer_origin[3]={
			paddingW*sizeof(float),
			paddingH,
			0};

	size_t host_origin[3]={0,
			0,
			0};

	size_t region[3]={d->width*sizeof(float),
			d->width,
			CHANNEL};

	size_t buffer_row_pitch= sizeof(float)*((d->width) + 2*paddingW );
	size_t buffer_slice_pitch=(d->width+2*paddingW)*(d->height+2*paddingH)*sizeof(float);

	size_t host_row_pitch=d->width*sizeof(float);
	size_t host_slice_pitch=(d->width)*(d->height)*sizeof(float);

	*(d->buffer) = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
			sizeof(float)
			* (WIDTH+2*paddingW)
			* (HEIGHT+2*paddingH)
			* CHANNEL, mem, &err);

	clEnqueueWriteBufferRect(
					queue,
					*(d->buffer),
					CL_TRUE,
					buffer_origin,
					host_origin,
					region,
					buffer_row_pitch,
					buffer_slice_pitch,
					host_row_pitch,
					host_slice_pitch,
					image,
					0,NULL, NULL);
	clFinish(queue);
	d->paddingW = paddingW;
	d->paddingH = paddingH;

}


double callKernel(float*images, float* filter, float*result, 
	int w, int h, int ITER, const char* KernelName, const char* KernelFunc)
{
	double st,et;
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_int err;

	cl_program program;
	FILE *program_handle;
	char *program_buffer, *program_log;
	size_t program_size, log_size;
	char options[256] = "";

	cl_kernel kernel;
	cl_mem images_buff;

	#if PRINT
	//writeFile(d, W, H, "d.csv");
	#endif

	clGetPlatformIDs(1, &platform, NULL);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

	program_handle = fopen(KernelName, "r");
	fseek(program_handle, 0, SEEK_END);
	program_size = ftell(program_handle);
	rewind(program_handle);
	program_buffer = (char*)malloc(program_size + 1);
	program_buffer[program_size] = '\0';
	fread(program_buffer, sizeof(char), program_size, program_handle);
	fclose(program_handle);

	program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer,
		&program_size, &err);

	free(program_buffer);

	sprintf(options,
		"-DCHANNELNUM=%d -DFILTERNUM=%d -DWIDTH=%d -DHEIGHT=%d -DWSTEP=%d -DHSTEP=%d",
		CHANNEL,FILTERNUM,WIDTH,HEIGHT,WSTEP,HSTEP);

	err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
	if(err < 0)
	{
		clGetProgramBuildInfo(program, device,CL_PROGRAM_BUILD_LOG,0,
			NULL,&log_size);
		program_log = (char*)malloc(sizeof(char)*(log_size+1));
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, device,
			CL_PROGRAM_BUILD_LOG,log_size+1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	kernel = clCreateKernel(program, KernelFunc, &err); 
	queue = clCreateCommandQueue(context, device, 0, &err);

	// program start
	
	imgData d;
	d.width = WIDTH;
	d.height = HEIGHT;
	d.paddingW = 0;
	d.paddingH = 0;
	d.buffer = &images_buff;

	float* mem = (float*)_mm_malloc(sizeof(float)*(WIDTH+6)*(HEIGHT+6)*CHANNEL,64);
	memset(mem, 0, sizeof(float)*(WIDTH+6)*(HEIGHT+6)*CHANNEL);
	st = second();

	memReshape(&d, mem,images, 2, 2, queue, context);

	// int W = d.width+d.paddingW*2;
	// int H = d.height+d.paddingH*2;

	// clSetKernelArg(kernel, 0, sizeof(cl_mem), d.buffer);
	// clSetKernelArg(kernel, 1, sizeof(int), &W);
	// clSetKernelArg(kernel, 2, sizeof(int), &H);
	
	// size_t global_work_size=1;
	// size_t local_work_size=1;

	// clEnqueueNDRangeKernel(queue, kernel, 1, NULL, 
	// 			&global_work_size, &local_work_size, 0, NULL, NULL);

	clFinish(queue);
	
	et = second();
	_mm_free(mem);
	clReleaseMemObject(*(d.buffer));
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);
	return et-st;
}
