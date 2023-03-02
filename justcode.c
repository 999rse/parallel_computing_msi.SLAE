#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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
        for (i = 0; i < N; i++) {
            norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        if (sqrt(norm) < epsilon) {
            break;
        }
        // обновление вектора x
        for (i = 0; i < N; i++) {
            x[i] = x_new[i];
        }
    }
    //for (i = 0; i < N; i++)
    //{
    //    printf("x[%d] = %f\n", i, x[i]);
    //}   
}
int main(){
    double A[N][N];
    double b[N];
    double x[N];
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
    simple_iteration_method(A,b,x,0.0001,10e6);
    clock_t end = clock(); // Остановка таймер
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Reference\n");
    printf("x[%d] = %f\n", N-2, x[N-2]);
    printf("x[%d] = %f\n", N-1, x[N-1]);
    printf("The elapsed time is %f seconds\n", time_spent);
}
