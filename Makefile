CC=g++
CFLAGS=-O0 -lOpenCL
INC=-I/opt/intel/opencl/include/
LIB=-L/opt/intel/opencl/
PROGS=conv
objs=main.o OCLKernel.o
# defines
# 
CFLAGS+= -DCHANNEL=32 -DFILTERNUM=32 -DWIDTH=356 -DHEIGHT=212 -DWSTEP=6 -DHSTEP=2

#CFLAGS+= -g

.c.o:
	$(CC) -c $< -o $@ $(CFLAGS) $(INC) $(LIB)

$(PROGS): $(objs)
	$(CC) $(objs) -o $@ $(CFLAGS) $(INC) $(LIB)

run: $(PROGS)
	./$(PROGS)

clean:
	rm -rf *.o *.ll *.csv $(PROGS)
gen:
	ioc64 -cmd=build -input=Kernel.cl -llvm-spir64=Kernel.ll
