#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define N 1000  // define matrix size
#define NUM_THREADS 4

double A[N][N];
double b[N];
double x[N];
double epsilon = 0.0001;
int max_iterations = 10e6;
int thread_num = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

void* simple_iteration_method(void* rank) {
    int i, j, k;
    int my_rank = *((int*) rank);
    int local_n = N / NUM_THREADS;
    int my_first_row = my_rank * local_n;
    int my_last_row = (my_rank + 1) * local_n - 1;
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
        // синхронизация потоков
        pthread_mutex_lock(&mutex);
        thread_num++;
        if (thread_num == NUM_THREADS) {
            thread_num = 0;
            pthread_cond_broadcast(&cond);
        } else {
            while (pthread_cond_wait(&cond, &mutex) != 0);
        }
        pthread_mutex_unlock(&mutex);
        // обновление вектора x
        for (i = my_first_row; i <= my_last_row; i++) {
            x[i] = x_new[i];
        }
        // проверка на сходимость
        pthread_mutex_lock(&mutex);
        double norm = 0;
        for (i = 0; i < N; i++) {
            norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
        }
        if (sqrt(norm) < epsilon) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}
int main() {
    int i, j;
    double start, end;
    pthread_t thread_handles[NUM_THREADS];
    int thread_ranks[NUM_THREADS];

    srand(time(NULL));  // initialize random number generator
    // initialize matrix A and vector b
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i != j) {
                A[i][j] = (double) rand() / RAND_MAX;
            }
        }
        b[i] = (double) rand() / RAND_MAX;
    }

    // create threads for simple iteration method
    start = clock();
    for (i = 0; i < NUM_THREADS; i++) {
        thread_ranks[i] = i;
        pthread_create(&thread_handles[i], NULL, simple_iteration_method, (void*) &thread_ranks[i]);
    }
    
    // join threads
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(thread_handles[i], NULL);
    }
    end = clock();
    printf("Pthreads-C\n");
    // print solution vector x
    printf("Solution vector x:\n");
    for (i = N-2; i < N; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    printf("Elapsed time = %.8f seconds\n", (end - start) / CLOCKS_PER_SEC);
    return 0;
}
