#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define N 1000  // define matrix size

void simple_iteration_method(double A[N][N], double b[N], double x[N], double epsilon, int max_iterations) {
    int i, j, k;
    double x_new[N];
    for (i = 0; i < N; i++) {
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
        #pragma omp parallel for shared(x_new, x, A, b) private(i, j)
        for (i = 0; i < N; i++) {
            double sum = 0;
            for (j = 0; j < N; j++) {
                if (i != j) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }
        // проверка на сходимость
        double norm = 0;
        #pragma omp parallel for reduction(+: norm) shared(x_new, x)
        for (i = 0; i < N; i++) {
            norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        if (sqrt(norm) < epsilon) {
            break;
        }
        // обновление вектора x
        #pragma omp parallel for shared(x_new, x)
        for (i = 0; i < N; i++) {
            x[i] = x_new[i];
        }
    }
    //for (i = 0; i < N; i++)
    //{
    //    printf("x[%d] = %f\n", i, x[i]);
    //}   
}
int main() {
    int i, j;
    double A[N][N], b[N], x[N], epsilon = 1e-6;
    int max_iterations = 1000;

    // initialize A, b, and x
    srand(time(NULL));
    for (i = 0; i < N; i++) {
        b[i] = (double)rand() / RAND_MAX;
        x[i] = 0;
        for (j = 0; j < N; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // measure the execution time of the simple iteration method
    double start_time = omp_get_wtime();
    simple_iteration_method(A, b, x, epsilon, max_iterations);
    double end_time = omp_get_wtime();
    printf("OpenMP-C\n");
    printf("x[%d] = %f\n", N-2, x[N-2]);
    printf("x[%d] = %f\n", N-1, x[N-1]);
    printf("The elapsed time is %f seconds\n", time_spent);
    return 0;
}
