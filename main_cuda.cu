#include <iostream>
#include <stdlib.h>
#include <cmath>
#include "boundary.h"
#include "jacobi.h"
#include "cfdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <numeric>
// using namespace std;

__global__ void sum(float* input)
{
	const int tid = threadIdx.x;
    input[tid] = input[tid] * input[tid];
	auto step_size = 1;
	int number_of_threads = blockDim.x;

	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) // still alive?
		{
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] += input[snd];
		}

		step_size <<= 1; 
		number_of_threads >>= 1;
	}
}
int main(int argc, char **argv) {
    int printfreq = 1000;
    float error, bnorm;

    //main arrays
    float *psi;
    //temp versions of main array
    float *psitmp;

    //comman line args
    int scalefactor, numiter;

    //simulation sizes
    int bbase = 10;
    int hbase = 10;
    int wbase = 5;
    int mbase = 32;
    int nbase = 32;

    int m, n, b, h, w;
    int iter;

    if (argc != 3) {
         std::cout << "Usage: cfd-cuda <scale> <numiter>\n";
    }

    scalefactor = atoi(argv[1]);
    numiter = atoi(argv[2]);

    std::cout << "Scale Factor = " << scalefactor << ", iterations " << numiter << "\n";

    b = bbase * scalefactor;
    h = hbase * scalefactor;
    w = wbase * scalefactor;
    m = mbase * scalefactor;
    n = nbase * scalefactor;

    std::cout << "Running CFD on" << m << " x " << n << " grid.\n";

    psi = new float[(m + 2) * (n + 2)];
    psitmp = new float[(m + 2) * (n + 2)];

    for (int i = 0; i < (m + 2) * (n + 2); i++) {
        psi[i] = 0;
    }

    //set the psi boundary conditions
    boundarypsi(psi, m, n, b, h, w);

    //compute normalization factor for error
    bnorm = 0;

    // can be parallelised like sum-reduction maybe
    // for (int i = 0; i < (m + 2) * (n + 2); i++) {
    //     bnorm += psi[i] * psi[i];
    // }
    // bnorm = std::sqrt(bnorm);
    
    // Cuda code
    float *psi_c;
    int count = (m+2)*(n+2);
    int size = count * sizeof(float);
    cudaMalloc(&psi_c, size);
    cudaMemcpy(psi_c, psi, size, cudaMemcpyHostToDevice);
    sum<<<1, count / 2>>>(psi_c);
    cudaMemcpy(&bnorm, psi_c, sizeof(float), cudaMemcpyDeviceToHost);
    bnorm = std::sqrt(bnorm);
    cudaFree(psi_c);

    // begin iterative jacobi loop
    std::cout << "Starting main loop...\n\n";

    for (iter = 1; iter <= numiter; iter++) {
        //calculate psi for next iteration
        jacobistep(psitmp, psi, m, n);

        if (iter == numiter) {
            error = deltasq(psitmp, psi, m, n);
            error = std::sqrt(error);
            error = error / bnorm;
        }

        //copy back
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= m; j++) {
                psi[i * (m + 2) + j] = psitmp[i * (m + 2) + j];
            }
        }

        //print loop info
        if (iter % printfreq == 0) {
            std::cout << "Completed iteration " << iter << "\n";
        }
    }

    if (iter > numiter)iter = numiter;
    std::cout << "\n...finished\n";
    std::cout << "After " << iter << " iterations, the error is " << error << "\n";

    //write output files

    writedatafiles(psi, m, n, scalefactor);
    writeplotfile(m, n, scalefactor);

    //fre un-needed arrays
    delete[] psi;
    delete[] psitmp;

    return 0;
}