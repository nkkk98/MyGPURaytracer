#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = idata[0];
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i];
            }
            for (int i = 0; i < n; i++)
            {
                odata[i] -= idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int* odata, const int* idata) {
            timer().startCpuTimer();
            // TODO
            int num = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0) {
                    odata[num++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return num;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int* odata, const int* idata) {
            int* tmpArray = new int[n];
            int* scanResult = new int[n];
            timer().startCpuTimer();
            // TODO
            //Step1.map
            for (int i = 0; i < n; i++)
            {
                if (idata[i] == 0) {
                    tmpArray[i] = 0;
                }
                else {
                    tmpArray[i] = 1;
                }
            }
            //Step2.scan
            scanResult[0] = tmpArray[0];
            for (int i = 1; i < n; i++) {
                scanResult[i] = scanResult[i - 1] + tmpArray[i];
            }
            for (int i = 0; i < n; i++)
            {
                scanResult[i] -= tmpArray[i];
            }
            //Step3.Scatter
            int num = 0;
            for (int i = 0; i < n; i++)
            {
                if (tmpArray[i]==1) {
                    odata[scanResult[i]] = idata[i];
                    num++;
                }
            }
            timer().endCpuTimer();
            delete []tmpArray;
            delete []scanResult;
            return num;
        }
    }
}
