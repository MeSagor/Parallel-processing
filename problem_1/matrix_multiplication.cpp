#include <iostream>
#include <vector>
#include <mpi.h>

using namespace std;

// Function to perform matrix multiplication
vector<int> matrixMultiplication(const int* A, const int* B, int M, int N, int P) {
    vector<int> result(M * P, 0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            for (int k = 0; k < N; ++k) {
                result[i * P + j] += A[i * N + k] * B[k * P + j];
            }
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    const int K = 5000; // Number of matrices
    const int M = 20;  // Rows of A
    const int N = 20;  // Columns of A, Rows of B
    const int P = 20;  // Columns of B

    vector<int> matricesA(K * M * N);
    vector<int> localA(K / numProcesses * M * N);
    vector<int> matricesB(K * N * P);
    vector<int> localB(K / numProcesses * N * P);
    vector<int> resultMatrix(K * M * P);
    vector<int> localResult(K / numProcesses * M * P);

    if (numProcesses < 1) {
        cerr << "This program requires at least 1 process." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (numProcesses > K) {
        cerr << "Number of processes cannot exceed the number of matrices (K)." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        // Initialize matrices A and B in the master process (rank 0)
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < M * N; ++j) {
                matricesA[i * M * N + j] = 2; // Example initialization
            }
        }
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < M * N; ++j) {
                matricesB[i * N * P + j] = 2; // Example initialization
            }
        }
    }

    clock_t start1, end1;
    start1 = clock();

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Scatter matrices A to all processes
    MPI_Scatter(&matricesA[0], M * N * K / numProcesses, MPI_INT,
        &localA[0], M * N * K / numProcesses, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter matrices B to all processes
    MPI_Scatter(&matricesB[0], N * P * K / numProcesses, MPI_INT,
        &localB[0], N * P * K / numProcesses, MPI_INT, 0, MPI_COMM_WORLD);


    // Perform matrix multiplication in each process
    for (int i = 0; i < K / numProcesses; ++i) {
        vector<int> tempResult = matrixMultiplication(&localA[i * M * N], &localB[i * N * P], M, N, P);
        // Copy tempResult to the appropriate section of localResult
        for (int j = 0; j < M * P; ++j) {
            localResult[i * M * P + j] = tempResult[j];
        }
    }

    // Gather localResult from all processes to the master process
    MPI_Gather(&localResult[0], M * P * K / numProcesses, MPI_INT,
        &resultMatrix[0], M * P * K / numProcesses, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);

    if (rank == 0) {
        cout << "Total time [ProcessNo: " << numProcesses << "]-> " << end - start << endl;
        // // Print the result matrices
        // cout << "Result Matrices:" << endl;
        // for (int i = 0; i < K; ++i) {
        //     cout << "Matrix " << i << ":" << endl;
        //     for (int row = 0; row < M; ++row) {
        //         for (int col = 0; col < P; ++col) {
        //             cout << resultMatrix[i * M * P + row * P + col] << " ";
        //         }
        //         cout << endl;
        //     }
        //     cout << endl;
        // }
    }

    MPI_Finalize();

    return 0;
}
