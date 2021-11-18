#pragma once

#include <vector>
#include "scene.h"
#include "timer.h"
PerformanceTimer& timer();
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void sendToGPU(uchar4* pbo,int iter);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);