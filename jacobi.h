#pragma once

void jacobistep(double *psinew, double *psi, int m, int n);

double deltasq(double *newarr, double *oldarr, int m, int n);

void jacobiiter_gpu(double *psi, int m, int n, int numiter, double &error);

__global__ void jacobikernel(double *psi_d, double *psinew_d, int m, int n, int numiter);