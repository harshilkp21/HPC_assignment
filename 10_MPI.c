#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int numbers[2];
    MPI_Status status;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get rank and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process creates its two numbers
    numbers[0] = rank * 2 + 1;     // example number 1
    numbers[1] = rank * 2 + 2;     // example number 2

    // Print what each process has
    printf("Process %d has numbers: %d and %d\n", rank, numbers[0], numbers[1]);

    // If not root, send two numbers to root (rank 0)
    if (rank != 0) {
        MPI_Send(numbers, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } 
    else {
        // Root process prints its own numbers
        printf("Root process received from itself: %d and %d\n", numbers[0], numbers[1]);

        // Root receives from all other processes
        for (int i = 1; i < size; i++) {
            int recv_nums[2];
            MPI_Recv(recv_nums, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            printf("Root received from process %d: %d and %d\n", i, recv_nums[0], recv_nums[1]);
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
