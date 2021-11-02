#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernUpSweep(int n, int d, int* Array) {
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x)<<(d+1);
            if (index < n) {
                Array[index +( 1 << (d + 1) )- 1] += Array[index+(1<<d)-1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* Array) {
            int index = ((blockIdx.x * blockDim.x) + threadIdx.x)<<(d+1);
            if (index < n) {
                int t = Array[index + (1 << d) - 1];
                Array[index +( 1 << d) - 1] = Array[index + (1 << (d + 1)) - 1];
                Array[index + (1 << (d + 1)) - 1] += t;
            }
        }
        __global__ void kernSetRootZero(int n, int* Array) {
            Array[n - 1] = 0;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int d = ilog2ceil(n);
            int paddedN = 1 << d;
            int* binaryTree;
            cudaMalloc((void**)&binaryTree, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc binaryTree failed!");
            cudaMemcpy(binaryTree, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy binaryTree failed!");

            timer().startGpuTimer();
            // TODO
            for (int i = 0; i < d; i++)
            {
                int threads = paddedN >> (i + 1);
                dim3 fullBlocksPerGrid((threads + blockSize - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (paddedN, i, binaryTree);
            }
            
            int rootValue = 0;
            kernSetRootZero << <1, 1 >> > (paddedN, binaryTree);
            
            for (int i = d - 1; i >= 0; i--)
            {
                int threads=paddedN >> (i + 1);
                dim3 fullBlocksPerGrid((threads + blockSize - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (paddedN, i, binaryTree);
            }
            
            timer().endGpuTimer();
            cudaMemcpy(odata, binaryTree, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("d2h cudaMemcoy failed!");
            cudaFree(binaryTree);
            checkCUDAError("cudaFree binaryTree1 failed!");
        }
        
        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int d = ilog2ceil(n);
            int paddedN = 1 << d;
            int* tmpArray;
            cudaMalloc((void**)&tmpArray, paddedN*sizeof(int));
            checkCUDAError("cudaMalloc failed!");
            int* dev_data;
            cudaMalloc((void**)&dev_data, paddedN*sizeof(int));
            checkCUDAError("cudaMalloc failed!");
            cudaMemset(dev_data, 0, paddedN*sizeof(int));
            cudaMemcpy(dev_data, idata, n*sizeof(int), cudaMemcpyHostToDevice);

            int* binaryTree;
            cudaMalloc((void**)&binaryTree, paddedN * sizeof(int));
            checkCUDAError("cudaMalloc binaryTree failed!");

            int* dev_data_out;
            cudaMalloc((void**)&dev_data_out, n * sizeof(int));

            dim3 fullBlocksPerGridOrigin((paddedN + blockSize - 1) / blockSize);
            timer().startGpuTimer();
            // TODO
            //Step1. Map
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGridOrigin, blockSize >> > (paddedN, tmpArray, dev_data);
            //Step2. Scan
            cudaMemcpy(binaryTree, tmpArray, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy binaryTree failed!");
            for (int i = 0; i < d; i++)
            {
                int threads = paddedN >> (i + 1);
                dim3 fullBlocksPerGrid((threads + blockSize - 1) / blockSize);
                kernUpSweep << <fullBlocksPerGrid, blockSize >> > (paddedN, i, binaryTree);
            }

            int rootValue = 0;
            kernSetRootZero << <1, 1 >> > (paddedN, binaryTree);

            for (int i = d - 1; i >= 0; i--)
            {
                int threads = paddedN >> (i + 1);
                dim3 fullBlocksPerGrid((threads + blockSize - 1) / blockSize);
                kernDownSweep << <fullBlocksPerGrid, blockSize >> > (paddedN, i, binaryTree);
            }
            //Step3. Scatter
            dim3 fullBlocksPerGridN((n + blockSize - 1) / blockSize);
            StreamCompaction::Common::kernScatter << <fullBlocksPerGridN, blockSize >> > (n, dev_data_out, dev_data, tmpArray, binaryTree);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_data_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            int count;
            cudaMemcpy(&count, &binaryTree[n-1], sizeof(int), cudaMemcpyDeviceToHost);
            int lastBool;
            cudaMemcpy(&lastBool, &tmpArray[n-1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(tmpArray);
            cudaFree(dev_data);
            cudaFree(binaryTree);
            cudaFree(dev_data_out);
            return count+lastBool;
        }
    }
}
