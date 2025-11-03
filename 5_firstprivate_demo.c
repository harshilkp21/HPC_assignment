#include <stdio.h>
#include <omp.h>

int main() {
    int val = 1234;
    printf("Initial value of val outside parallel region: %d\n", val);

    #pragma omp parallel num_threads(4) firstprivate(val)
    {
        int tid = omp_get_thread_num();
        printf("Thread %d: initial val = %d\n", tid, val);
        val++;
        printf("Thread %d: updated val = %d\n", tid, val);
    }

    printf("Final value of val outside parallel region: %d\n", val);
    return 0;
}
