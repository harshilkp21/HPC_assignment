#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int N = 10;                // Total number of integers (example)
    int local_value;           // Each process holds one number
    int sum;                   // Variable to carry partial sum
    int next, prev;
    MPI_Status status;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get rank (ID) and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process gets a number (for demo, use rank+1)
    local_value = rank + 1;

    // Set up ring topology connections
    next = (rank + 1) % size;              // next process in ring
    prev = (rank - 1 + size) % size;       // previous process in ring

    sum = local_value;

    // Start the ring-based sum communication
    for (int i = 0; i < size - 1; i++) {
        int received_value;

        // Send current sum to the next process
        MPI_Send(&sum, 1, MPI_INT, next, 0, MPI_COMM_WORLD);

        // Receive partial sum from previous process
        MPI_Recv(&received_value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);

        // Add received value to local sum
        sum = received_value + local_value;
    }

    // At the end of the ring, the process that started (rank 0) will have the total sum
    if (rank == 0) {
        printf("\nTotal sum of first %d integers = %d\n", size, sum);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
