all:
	gcc justcode.c -o justcode -lm
	gcc pthreadcode.c -o pthreadcode -lm
	mpicc mpicode.c -o mpicode -lm
	gcc openmpcode.c -o openmpcode -lm -fopenmp
