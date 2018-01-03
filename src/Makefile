add2array.o:	add2array.cu
	nvcc -c add2array.cu -o add2array.o

add2array:	add2array.o
	nvcc add2array.o -o add2array

all:	add2array

clean:
	rm -f *.o add2array
