#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define N 1000  // define matrix size

void simple_iteration_method(double A[N][N], double b[N], double x[N], double epsilon, int max_iterations, int numprocs, int myrank) {
    int i, j, k;
    int local_n = N / numprocs;
    int my_first_row = myrank * local_n;
    int my_last_row = (myrank + 1) * local_n - 1;
    double x_new[N];
    for (i = my_first_row; i <= my_last_row; i++) {
        double subsum = 0;
        for (j = 0; j < N; j++) {
            if (i != j) {
                subsum += fabs(A[i][j]);
            }
        }
        A[i][i] = 2.0 * subsum;
    }
    for (k = 0; k < max_iterations; k++) {
        // вычисление нового приближения x
        for (i = my_first_row; i <= my_last_row; i++) {
            double sum = 0;
            for (j = 0; j < N; j++) {
                if (i != j) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }
        // синхронизация процессов
        MPI_Allgather(MPI_IN_PLACE, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        // обновление вектора x
        for (i = my_first_row; i <= my_last_row; i++) {
            x[i] = x_new[i];
        }
        // проверка на сходимость
        double norm = 0;
        for (i = 0; i < N; i++) {
            norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        double global_norm = 0;
        MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (sqrt(global_norm) < epsilon) {
            break;
        }
    }
}
int main(int argc, char **argv) {
    double A[N][N];
    double b[N];
    double x[N];
    int numprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // для хранения времени выполнения кода
    double time_spent = 0.0;
    // Открытие файла на чтение матрицы А
    FILE *f = fopen("A.txt", "r");
    FILE *fb = fopen("b.txt", "r");
    // Чтение матрицы из файла
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(f, "%lf", &A[i][j]);
        }
        fscanf(fb,"%lf", &b[i]);
        x[i] = 0;
    }
    // Закрытие файла
    fclose(f);
    fclose(fb);
    clock_t begin = clock(); // Запуск таймер
    simple_iteration_method(A,b,x,0.0001,10e6,numprocs,myrank);
    clock_t end = clock(); // Остановка таймер
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    if (myrank == 0) {
	printf("MPI-C\n");
	for (int i = N-2; i < N; i++) {
            printf("x[%d] = %f\n", i, x[i]);
        }
        printf("The elapsed time is %f seconds\n", time_spent);
    }
    MPI_Finalize();
    return 0;
}
