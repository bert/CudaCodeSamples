all:	test

test:	main.o
  nvcc main.o -o test

main.o:	main.cu
  nvcc -c main.cu -o main.o

clean:
  rm -f *.o test
