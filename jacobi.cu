#include "jacobi.h"
#include <iostream>

__device__ double d_error;
const int SHMEM_SIZE = 1 << 10;
__global__ void jacobikernel(double *psi_d, double *psinew_d, int m, int n, int numiter) {

    // calculate each thread's global row and col
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int SHMEM_SIZE = sizeof(double) * (m + 2) * (n + 2);

    __shared__ int s_a[SHMEM_SIZE];
    s_a[row * (m + 2) + col] = psi_d[row * (m + 2) + col];

    if (row > 0 && row <= m && col > 0 && col <= n) {
        for (int i = 1; i <= numiter; i++) {
            d_error = 0;
            psinew_d[row * (m + 2) + col] =
                    0.25f * (s_a[(row - 1) * (m + 2) + col] + s_a[(row + 1) * (m + 2) + col] +
                             s_a[(row) * (m + 2) + col - 1] + s_a[(row) * (m + 2) + col + 1]);

            __syncthreads();
            s_a[row * (m + 2) + col] = psinew_d[row * (m + 2) + col];
            __syncthreads();
        }
    }
    psi_d[row * (m + 2) + col] = s_a[row * (m + 2) + col];
    __syncthreads();

}

//void jacobistep(double *psinew, double *psi, int m, int n) {
//    for (int i = 1; i <= m; i++) {
//        for (int j = 1; j <= n; j++) {
//            psinew[i * (m + 2) + j] = 0.25f * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] +
//                                               psi[(i) * (m + 2) + j - 1] + psi[(i) * (m + 2) + j + 1]);
//        }
//    }
//}

void jacobiiter_gpu(double *psi, double *psitmp, int m, int n, int numiter, double &error) {

//     double *psi_d;
//     double *psinew_d;
//     size_t bytes = sizeof(double) * (m + 2) * (n + 2);

//     // allocate memory on gpu
//     cudaMalloc(&psi_d, bytes);
//     cudaMalloc(&psinew_d, bytes);

//     // copy data to gpu
//     cudaMemcpy(psi_d, psi, bytes, cudaMemcpyHostToDevice);
// //    cudaMemcpy(psinew_d, psinew, bytes, cudaMemcpyHostToDevice);

    int THREADS = 16;
    int BLOCKS = (m + 2 + THREADS - 1) / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    jacobikernel<<<blocks, threads>>>(psi, psitmp, m, n, numiter);

    // cudaMemcpy(psi, psi_d, bytes, cudaMemcpyDeviceToHost);
//
//    for (int i = 0; i<(m+2)*(n+2); i++){
//        std::cout<<psi[i]<<" ";
//    }

//    double e;
//    cudaMemcpyFromSymbol(&e, "d_error", sizeof(e), 0, cudaMemcpyDeviceToHost);
//    error = e;

    // cudaFree(psi_d);
    // cudaFree(psinew_d);
}

// parallelise
void jacobistep(double *psinew, double *psi, int m, int n) {
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= m; j++) {
            psinew[i * (m + 2) + j] = 0.25f * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] +
                                               psi[(i) * (m + 2) + j - 1] + psi[(i) * (m + 2) + j + 1]);
        }
    }
}

// parallelise
double deltasq(double *newarr, double *oldarr, int m, int n) {
    double dsq = 0;
    double tmp;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= m; j++) {
            tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
            dsq += tmp * tmp;
        }
    }

    return dsq;
}