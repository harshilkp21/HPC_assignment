#include <stdio.h>
#include <cuda.h>

// CUDA kernel for matrix addition
__global__ void matrixAdd(int *A, int *B, int *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * cols + col;

    // Check bounds
    if (row < rows && col < cols) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int rows = 3, cols = 3;  // You can change matrix size here
    int size = rows * cols * sizeof(int);

    int A[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int B[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    int C[9];

    int *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // Launch the kernel
    matrixAdd<<<grid, block>>>(d_A, d_B, d_C, rows, cols);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("Matrix A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%d ", A[i * cols + j]);
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%d ", B[i * cols + j]);
        printf("\n");
    }

    printf("\nMatrix C (A + B):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%d ", C[i * cols + j]);
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
