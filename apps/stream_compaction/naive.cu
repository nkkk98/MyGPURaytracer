#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO:
        __global__ void inclusiveScan(int n, int d, int* oArray, int* iArray) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int gap = 1 << d;
            if (index < n) {
                if (index < gap)oArray[index] = iArray[index];
                else oArray[index] = iArray[index - gap] + iArray[index];
            }
        }
        __global__ void changeToExclusive(int n, int* oArray, int* iArray) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index == 0) {
                oArray[index] = 0;
            }else if (index < n) oArray[index] = iArray[index - 1];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {

            int depth = ilog2ceil(n);

            int* dev_data1; int* dev_data2;
            cudaMalloc((void**)&dev_data1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataExtended1 failed!");
            cudaMalloc((void**)&dev_data2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_dataExtended2 failed!");

            // copy idata to device memory 
            cudaMemcpy(dev_data1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data1 failed!");

            timer().startGpuTimer();
            // TODO
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
            //Step1: Cal Inclusive scan
            for (int d = 0; d < depth; d++)
            {
                inclusiveScan << <fullBlocksPerGrid, blockSize >> > (n, d, dev_data2, dev_data1);
                int* tmp = dev_data1;
                dev_data1 = dev_data2;
                dev_data2 = tmp;
            }
            //Step2: Change to Exclusive scan
            changeToExclusive << <fullBlocksPerGrid, blockSize >> > (n, dev_data2, dev_data1);
            cudaMemcpy(odata, dev_data2, sizeof(int)*n, cudaMemcpyDeviceToHost);
            timer().endGpuTimer();

            cudaFree(dev_data1);
            cudaFree(dev_data2);
        }
    }
}
