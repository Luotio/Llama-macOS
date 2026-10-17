#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"

int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    printf("device count: %d\n", n);
    for (int i = 0; i < n; i++) {
        cudaSetDevice(i);
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        size_t freeb = 0, totalb = 0;
        cudaMemGetInfo(&freeb, &totalb);
        printf("dev %d: %s, total=%.1f GiB, free=%.1f GiB\n",
               i, p.name, totalb / 1073741824.0, freeb / 1073741824.0);
        size_t sz = 144ull * 1024 * 1024;
        void *d = nullptr;
        cudaError_t e = cudaMalloc(&d, sz);
        printf("  cudaMalloc(144 MiB) -> %s\n", cudaGetErrorString(e));
        if (e == cudaSuccess) {
            e = cudaMemset(d, 0, sz);
            printf("  cudaMemset -> %s\n", cudaGetErrorString(e));
            e = cudaDeviceSynchronize();
            printf("  sync -> %s\n", cudaGetErrorString(e));
            cudaFree(d);
        }
    }
    return 0;
}
