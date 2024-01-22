
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int K = 10000;
const int M = 20;
const int N = 20;
const int P = 20;

__global__ void temp_matrixMultiplicationGPU(const int* A, const int* B, int* C, int M, int N, int P, int matricesPerThread) {
    int matrixIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (matrixIdx < K && row < M && col < P) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[matrixIdx * M * N + row * N + k] * B[matrixIdx * N * P + k * P + col];
        }
        C[matrixIdx * M * P + row * P + col] = sum;
    }
}

__global__ void matrixMultiplicationGPU(const int* A, const int* B, int* C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

void matrixMultiplicationCPU(const int* A, const int* B, int* C, int M, int N, int P) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

int main() {
    const int size_A = K * M * N;
    const int size_B = K * N * P;
    const int size_C = K * M * P;

    // Allocate host memory
    int* h_A = new int[size_A];
    int* h_B = new int[size_B];
    int* h_C_GPU = new int[size_C];
    int* h_C_CPU = new int[size_C];

    // Initialize matrices A and B on the host
    for (int i = 0; i < size_A; ++i) {
        h_A[i] = i % 10; // Example initialization
    }

    for (int i = 0; i < size_B; ++i) {
        h_B[i] = 1; // Example initialization
    }

    // Allocate device memory
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size_A * sizeof(int));
    cudaMalloc((void**)&d_B, size_B * sizeof(int));
    cudaMalloc((void**)&d_C, size_C * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(int), cudaMemcpyHostToDevice);

    // Create events for timing GPU multiplication
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    // Record start event for GPU
    cudaEventRecord(startGPU);

    // Set the block and grid dimensions
    // dim3 dimBlock(16, 16, 1);
    // dim3 dimGrid((P + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y, (K + dimBlock.z - 1) / dimBlock.z);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((P + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    // Launch the GPU kernel for each matrix multiplication
    matrixMultiplicationGPU<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, P);


    // Record stop event for GPU
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    // Calculate elapsed time for GPU
    float millisecondsGPU = 0;
    cudaEventElapsedTime(&millisecondsGPU, startGPU, stopGPU);

    // Copy the result matrix C from device to host for GPU
    cudaMemcpy(h_C_GPU, d_C, size_C * sizeof(int), cudaMemcpyDeviceToHost);

    // Record start time for CPU
    auto startCPU = std::chrono::high_resolution_clock::now();

    // Perform CPU matrix multiplication
    matrixMultiplicationCPU(h_A, h_B, h_C_CPU, M, N, P);

    // Record stop time for CPU
    auto stopCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(stopCPU - startCPU);

    // Print the elapsed time for GPU
    std::cout << "Time taken (GPU): " << millisecondsGPU << " microseconds" << std::endl;

    // Print the elapsed time for CPU
    std::cout << "Time taken (CPU): " << durationCPU.count() << " microseconds" << std::endl;

    // Verify GPU and CPU results
    for (int i = 0; i < size_C; ++i) {
        if (h_C_GPU[i] != h_C_CPU[i]) {
            std::cerr << "Verification failed!" << std::endl;
            break;
        }
    }

    // Free allocated memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_GPU;
    delete[] h_C_CPU;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // cudaEventDestroy(startGPU);
    // cudaEventDestroy(stopGPU);

    return 0;
}
