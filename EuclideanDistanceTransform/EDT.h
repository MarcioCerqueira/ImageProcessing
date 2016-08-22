#ifndef EDT_H
#define EDT_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

void GPUClearStructure(int *structure, int rows, int cols);
void GPUComputeNearestSiteInRow(unsigned char *image, int *nearestSite, int rows, int cols);
void GPUComputeProximateSitesInColumn(int *nearestSite, int *proximateSites, int rows, int cols);
void GPUComputeNearestSiteInFull(int *proximateSites, int *nearestSite, int rows, int cols);
void GPUComputeDistanceTransform(unsigned char *EDTImage, int *nearestSite, int rows, int cols);

#endif