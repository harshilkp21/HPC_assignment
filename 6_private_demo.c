#include <stdio.h>
#include <omp.h>

int main() {
    int val = 1234;  // Original variable

    // Print initial value outside the parallel region
    printf("Initial value of val outside parallel region: %d\n", val);

    // Parallel region with 4 threads using 'private' clause
    #pragma omp parallel num_threads(4) private(val)
    {
        int tid = omp_get_thread_num();

        // Each thread has its own private copy of 'val' (uninitialized)
        printf("Thread %d: initial val = %d (may be garbage)\n", tid, val);

        val++;  // Increment private copy

        printf("Thread %d: updated val = %d\n", tid, val);
    }

    // Print final value outside the parallel region
    printf("Final value of val outside parallel region: %d\n", val);

    return 0;
}

