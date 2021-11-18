#pragma once

#include <vector>
#include "scene.h"
#include "timer.h"
PerformanceTimer& timer();
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, bool denoise, int filterSize, float c_weight, float p_weight, float n_weight);
void sendToGPU(uchar4* pbo,int iter);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);