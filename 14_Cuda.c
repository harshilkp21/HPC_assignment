#include <stdio.h>
#include <cuda.h>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    int sum = 0;

    // Perform multiplication if within bounds
    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 3; // Size of NxN matrix (you can change it)
    int size = N * N * sizeof(int);

    // Host matrices
    int A[9] = {1, 2, 3,
                4, 5, 6,
                7, 8, 9};

    int B[9] = {9, 8, 7,
                6, 5, 4,
                3, 2, 1};

    int C[9]; // Result matrix

    // Device matrices
    int *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Launch the kernel
    matrixMultiply<<<grid, block>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print matrices
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", A[i * N + j]);
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", B[i * N + j]);
        printf("\n");
    }

    printf("\nMatrix C (A x B):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", C[i * N + j]);
        printf("\n");
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
