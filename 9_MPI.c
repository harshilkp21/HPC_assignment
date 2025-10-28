#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    char message[100];
    MPI_Status status;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process creates its message
    sprintf(message, "Hello World from process %d of %d", rank, size);
    printf("%s\n", message);

    // If not the root process, send message to root
    if (rank != 0) {
        MPI_Send(message, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } 
    // Root process receives messages from all other processes
    else {
        for (int i = 1; i < size; i++) {
            MPI_Recv(message, 100, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
            printf("Root received: %s\n", message);
        }
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}
