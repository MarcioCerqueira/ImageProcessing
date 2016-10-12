#include "EDT.h"

__device__ float GPUComputeEuclideanDistance(int sitePixel, int pixel, int cols) {

	int sx = sitePixel % cols;
	int sy = sitePixel / cols;
	int px = pixel % cols;
	int py = pixel / cols;
	return sqrtf((sx - px) * (sx - px) + (sy - py) * (sy - py));

}

__device__ bool GPUHasDomination(int a, int b, int c, int column, int cols) {

	float u, v;
	//p(i, u)
	int ax = a % cols;
	int ay = a / cols;
	int bx = b % cols;
	int by = b / cols;
	float mx = (float)(ax + bx) / 2;
	float my = (float)(ay + by) / 2;
	if(bx == ax) {
		u = my;
	} else if(by == ay) {
		u = my;
	} else {
		float m1 = (float)(by - ay) / (float)(bx - ax);
		float m2 = -1/m1;
		u = m2 * (column - mx) + my;
	}

	//q(i, v)
	int cx = c % cols;
	int cy = c / cols;
	mx = (float)(bx + cx) / 2;
	my = (float)(by + cy) / 2;
	if(cx == bx) {
		v = my;
	} else if(cy == by) {
		v = my;
	} else {
		float m1 = (float)(cy - by) / (float)(cx - bx);
		float m2 = -1/m1;
		v = m2 * (column - mx) + my;
	}
	
	if(u > v) return true;
	else return false;

}

__global__ void clearStructure(int *structure) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	structure[id] = -1;

}

__global__ void computeSelfNearestSite(unsigned char *image, int *nearestSite) {
	
	int pixel = blockIdx.x * blockDim.x + threadIdx.x;
	if(image[pixel] == 0) nearestSite[pixel] = pixel;

}

__global__ void computeNearestSiteInRow(unsigned char *image, int *nearestSite, int cols, int bandSize) {
	
	int x = blockIdx.x * bandSize;
	int y = threadIdx.x;
	for(int xp = x; xp < x + bandSize; xp++) {
	
		int pixel = y * cols + xp;
		if(image[pixel] == 0) {

			for(int xs = xp + 1; xs < x + bandSize; xs++) {
			
				int propagationPixel = y * cols + xs;
				if(image[propagationPixel] != 0) nearestSite[propagationPixel] = pixel;
				else break;
	
			}

		}
	
	}

	for(int xp = x + bandSize - 1; xp >= x; xp--) {
	
		int pixel = y * cols + xp;
		if(image[pixel] == 0) {

			for(int xs = xp - 1; xs >= x; xs--) {
			
				int propagationPixel = y * cols + xs;
				int imagePropagationPixel = image[propagationPixel];
				int nearestSitePropagationPixel = nearestSite[propagationPixel];
				if(nearestSitePropagationPixel == -1) {
					nearestSite[propagationPixel] = pixel;
				} else if(imagePropagationPixel != 0 && nearestSitePropagationPixel != -1) {
					float a = abs(nearestSitePropagationPixel % cols - xs);
					float b = abs(xp - xs);
					if(b < a) nearestSite[propagationPixel] = pixel;
				} else break;

			}

		}

	}
	
}

__global__ void updateBandSitesInRow(int *nearestSite, int cols, int bandSize, int iteration) {

	int x = blockIdx.x * bandSize + ((iteration + 1) % 2) * (bandSize - 1);
	int y = threadIdx.x;
	int pixel = y * cols + x;
	int neighbourPixel;

	if(iteration % 2 == 0) neighbourPixel = pixel + 1;
	else neighbourPixel = pixel + bandSize - 1;

	int nearestPixel1 = nearestSite[pixel];
	int nearestPixel2 = nearestSite[neighbourPixel];
	float a = GPUComputeEuclideanDistance(pixel, nearestPixel1, cols);
	float b = GPUComputeEuclideanDistance(pixel, nearestPixel2, cols);
	float c = GPUComputeEuclideanDistance(neighbourPixel, nearestPixel1, cols);
	float d = GPUComputeEuclideanDistance(neighbourPixel, nearestPixel2, cols);
	if(a > b) nearestSite[pixel] = nearestPixel2;
	if(d > c) nearestSite[neighbourPixel] = nearestPixel1;
	
}

__global__ void updateNearestSiteInRow(int *nearestSite, int cols, int bandSize) {
	
	int pixel = blockIdx.x * blockDim.x + threadIdx.x;
	int x = pixel % cols;
	int y = pixel / cols;
	int band = x / bandSize;
	int firstBandPixel = y * cols + band * bandSize;
	int lastBandPixel = y * cols + band * bandSize + bandSize - 1;
	float a = GPUComputeEuclideanDistance(pixel, nearestSite[pixel], cols);
	float b = GPUComputeEuclideanDistance(pixel, nearestSite[firstBandPixel], cols);
	float c = GPUComputeEuclideanDistance(pixel, nearestSite[lastBandPixel], cols);
	if(b < a && b <= c) nearestSite[pixel] = nearestSite[firstBandPixel];
	if(c < b && c < a) nearestSite[pixel] = nearestSite[lastBandPixel];
	
}

__global__ void computeProximateSitesInColumn(int *nearestSite, int *proximateSites, int rows, int cols, int bandSize) {

	//Here, our stack begins in "y + bandSize - 1" and ends in "y"
	int x = threadIdx.x;
	int y = blockIdx.x * bandSize;
	int count = y;

	for(int yb = y; yb < y + bandSize; yb++) {

		int pixel = yb * cols + x;
		int c = nearestSite[pixel];
		if(c != -1) {

			while(count >= y + 2) {
					
				int a = proximateSites[(count - 2) * cols + x];
				int b = proximateSites[(count - 1) * cols + x];
				if(GPUHasDomination(a, b, c, x, cols)) {
					proximateSites[(count - 1) * cols + x] = -1;
					count--;
				} else break;

			}
				
			proximateSites[count * cols + x] = c;
			count++;

		}

	}

}

__global__ void mergeProximateSitesInColumn(int *nearestSite, int *proximateSites, int rows, int cols, int bandSize) {

	int x = threadIdx.x;
	int count = 0;
	
	for(int y = 0; y < bandSize; y++)
		if(proximateSites[y * cols + x] != -1) count++;	
	
	for(int it = 1; it < rows/bandSize; it++) {
		int bandCount = 0;
		for(int y = 0; y < bandSize; y++) {

			int yp = y + it * bandSize;
			int pixel = yp * cols + x;
			int c = proximateSites[pixel];
			if(c != -1) {
			
				if(bandCount == 2) {
					proximateSites[count * cols + x] = c;
					count++;
					continue;
				}

				while(count >= 2) {
					
					int a = proximateSites[(count - 2) * cols + x];
					int b = proximateSites[(count - 1) * cols + x];
					if(GPUHasDomination(a, b, c, x, cols)) {
						proximateSites[(count - 1) * cols + x] = -1;
						count--;
						bandCount = 0;
					} else break;

				}
			
				proximateSites[count * cols + x] = c;
				count++;
				bandCount++;
				
			}

		}
	}

}

__global__ void computeNearestSiteInFullKernel(int *proximateSites, int *nearestSite, int rows, int cols) {

	int x = threadIdx.x;
	int count = 0;

	for(int y = 0; y < rows; y++) {

		int pixel = y * cols + x;
		while(count < rows - 1) {

			float a = GPUComputeEuclideanDistance(proximateSites[count * cols + x], pixel, cols);
			float b = GPUComputeEuclideanDistance(proximateSites[(count + 1) * cols + x], pixel, cols);
			if(a <= b) break;
			else count++;
				
				
		}
			
		nearestSite[pixel] = proximateSites[count * cols + x];
			
	}
	
}

__global__ void computeDistanceTransform(unsigned char *EDTImage, int *nearestSite, int cols) {
	
	int pixel = blockIdx.x * blockDim.x + threadIdx.x;
	EDTImage[pixel] = GPUComputeEuclideanDistance(nearestSite[pixel], pixel, cols);

}

void GPUCheckError(char *methodName) {

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) printf("%s: %s\n", methodName, cudaGetErrorString(error));
	
}

void GPUClearStructure(int *structure, int rows, int cols) {

    clearStructure<<<rows, cols>>>(structure);
	GPUCheckError("GPUClearStructure");

}

void GPUComputeNearestSiteInRow(unsigned char *image, int *nearestSite, int rows, int cols) {

	int bands = 16;
	int bandSize = cols / bands;
	computeSelfNearestSite<<<rows, cols>>>(image, nearestSite);
	computeNearestSiteInRow<<<bands, rows>>>(image, nearestSite, cols, bandSize);
	for(int it = 0; it < bands; it++) updateBandSitesInRow<<<bands - (int)((it + 1) % 2), rows>>>(nearestSite, cols, bandSize, it);
	updateNearestSiteInRow<<<rows, cols>>>(nearestSite, cols, bandSize);
	GPUCheckError("GPUComputeNearestSiteInRow");

}

void GPUComputeProximateSitesInColumn(int *nearestSite, int *proximateSites, int rows, int cols) {

	int bands = 16;
	int bandSize = rows / bands;
	computeProximateSitesInColumn<<<bands, cols>>>(nearestSite, proximateSites, rows, cols, bandSize);
	mergeProximateSitesInColumn<<<1, cols>>>(nearestSite, proximateSites, rows, cols, bandSize);
	GPUCheckError("GPUComputeProximateSitesInColumn");

}

void GPUComputeNearestSiteInFull(int *proximateSites, int *nearestSite, int rows, int cols) {

	computeNearestSiteInFullKernel<<<1, cols>>>(proximateSites, nearestSite, rows, cols);
	GPUCheckError("GPUComputeNearestSiteInFull");
	
}

void GPUComputeDistanceTransform(unsigned char *EDTImage, int *nearestSite, int rows, int cols) {

	computeDistanceTransform<<<rows, cols>>>(EDTImage, nearestSite, cols);
	GPUCheckError("GPUComputeDistanceTransform");

}
