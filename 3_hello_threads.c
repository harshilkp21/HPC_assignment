#include <stdio.h>
#include <omp.h>

int main() {
    // Set number of threads
    omp_set_num_threads(4);

    // Parallel region
    #pragma omp parallel
    {
        int tid = omp_get_thread_num(); // get thread ID
        printf("Hello World from thread %d\n", tid);
    }

    return 0;
}

