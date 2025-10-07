#include <stdio.h>
#include <omp.h>

#define N 20     // total loop iterations
#define CHUNK 3  // chunk size

int main() {
    omp_set_num_threads(4);

    #pragma omp parallel for schedule(static, CHUNK)
    for (int i = 0; i < N; i++) {
        printf("Thread %d is executing iteration %d\n", omp_get_thread_num(), i);
    }

    return 0;
}

