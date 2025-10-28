#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    long long int N = 10000;  // Total numbers to sum
    long long int local_start, local_end;
    long long int local_sum = 0, total_sum = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide N numbers equally among processes
    long long int numbers_per_process = N / size;
    local_start = rank * numbers_per_process + 1;
    local_end = (rank + 1) * numbers_per_process;

    // If last process, include any remaining numbers
    if (rank == size - 1)
        local_end = N;

    // Compute local sum
    for (long long int i = local_start; i <= local_end; i++) {
        local_sum += i;
    }

    printf("Process %d computed partial sum: %lld (from %lld to %lld)\n",
           rank, local_sum, local_start, local_end);

    // Reduce all partial sums to root process
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root prints final result
    if (rank == 0) {
        printf("\nTotal sum of first %lld integers = %lld\n", N, total_sum);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
