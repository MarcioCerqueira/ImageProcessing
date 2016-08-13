#include <opencv2\opencv.hpp>
#include <iostream>
#include <time.h>

double cpu_time(void)
{

	double value;
	value = (double) clock () / (double) CLOCKS_PER_SEC;
	return value;

}

void clearStructure(int *structure, int size) {

	for(int pixel = 0; pixel < size; pixel++)
		structure[pixel] = -1;

}

float computeEuclideanDistance(int sitePixel, int pixel, int imageCols) {

	int sx = sitePixel % imageCols;
	int sy = sitePixel / imageCols;
	int px = pixel % imageCols;
	int py = pixel / imageCols;
	return sqrtf(powf(sx - px, 2) + powf(sy - py, 2));

}
bool hasDomination(int a, int b, int c, int column, int imageCols) {

	int u, v;
	//p(i, u)
	int ax = a % imageCols;
	int ay = a / imageCols;
	int bx = b % imageCols;
	int by = b / imageCols;
	int mx = (ax + bx) / 2;
	int my = (ay + by) / 2;
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
	int cx = c % imageCols;
	int cy = c / imageCols;
	mx = (bx + cx) / 2;
	my = (by + cy) / 2;
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

void computeNearestSiteInRow(cv::Mat image, int *nearestSite) {
	
	//Every site is its own nearest site
	for(int y = 0; y < image.rows; y++) {
		for(int x = 0; x < image.cols; x++) {
			int sitePixel = y * image.cols + x;
			if(image.ptr<unsigned char>()[sitePixel] == 0) 
				nearestSite[sitePixel] = sitePixel;
		}
	}
		
	//Left to right sweep
	for(int y = 0; y < image.rows; y++) {
		for(int x = 0; x < image.cols; x++) {

			int sitePixel = y * image.cols + x;
			if(image.ptr<unsigned char>()[sitePixel] == 0) {
				
				for(int xs = x + 1; xs < image.cols; xs++) {
					
					int propagationPixel = y * image.cols + xs;
					if(image.ptr<unsigned char>()[propagationPixel] != 0) nearestSite[propagationPixel] = sitePixel;
					else break;

				}

			}

		}

	}

	//Right to left sweep
	for(int y = 0; y < image.rows; y++) {
		for(int x = image.cols - 1; x >= 0; x--) {

			int sitePixel = y * image.cols + x;
			if(image.ptr<unsigned char>()[sitePixel] == 0) {
				
				for(int xs = x - 1; xs >= 0; xs--) {
					
					int propagationPixel = y * image.cols + xs;
					if(image.ptr<unsigned char>()[propagationPixel] != 0 && nearestSite[propagationPixel] == -1) {
						nearestSite[propagationPixel] = sitePixel;
					} else if(image.ptr<unsigned char>()[propagationPixel] != 0 && nearestSite[propagationPixel] != -1) {
						float previousEuclideanDistance = computeEuclideanDistance(nearestSite[propagationPixel], propagationPixel, image.cols);
						float currentEuclideanDistance = sqrtf(powf(x - xs, 2) + powf(y - y, 2));
						if(currentEuclideanDistance < previousEuclideanDistance) nearestSite[propagationPixel] = sitePixel;
					} else break;

				}

			}

		}

	}

}

void computeProximateSitesInColumn(int *nearestSite, int *proximateSites, int imageRows, int imageCols) {
	
	//Here, our stack begins in "count" and ends in "0"
	int count, proximateIndex;

	for(int x = 0; x < imageCols; x++) {
		
		count = 0;

		for(int y = 0; y < imageRows; y++) {

			int pixel = y * imageCols + x;
			if(nearestSite[pixel] != -1) {

				int c = nearestSite[pixel];
				
				while(count >= 2) {
					
					int a = proximateSites[(count - 2) * imageCols + x];
					int b = proximateSites[(count - 1) * imageCols + x];
					if(hasDomination(a, b, c, x, imageCols)) {
						proximateSites[(count - 1) * imageCols + x] = -1;
						count--;
					} else break;

				}
				
				proximateSites[count * imageCols + x] = c;
				count++;

			}

		}

	}

}

void computeNearestSiteInFull(int *proximateSites, int *nearestSite, int imageRows, int imageCols) {

	int count;
	for(int x = 0; x < imageCols; x++) {
		count = 0;
		for(int y = 0; y < imageRows; y++) {

			int pixel = y * imageCols + x;
			while(count < imageRows - 1) {

				float a = computeEuclideanDistance(proximateSites[count * imageCols + x], pixel, imageCols);
				float b = computeEuclideanDistance(proximateSites[(count + 1) * imageCols + x], pixel, imageCols);
				if(a <= b) break;
				else count++;
				
				
			}
			
			nearestSite[pixel] = proximateSites[count * imageCols + x];
			
		}
	}

}

void computeDistanceTransform(cv::Mat EDTImage, int *nearestSite, int imageRows, int imageCols) {

	for(int y = 0; y < imageRows; y++) {
		for(int x = 0; x < imageCols; x++) {

			int pixel = y * imageCols + x;
			EDTImage.ptr<unsigned char>()[pixel] = computeEuclideanDistance(nearestSite[pixel], pixel, imageCols);

		}
	}

}

int main(int argc, char **argv) {
	
	cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat sites = cv::Mat(image.rows, image.cols, image.type());
	cv::Mat EDTImage = cv::Mat(image.rows, image.cols, image.type());
	int *nearestSite = (int*)malloc(image.rows * image.cols * sizeof(int));
	int *proximateSites = (int*)malloc(image.rows * image.cols * sizeof(int));

	double begin = cpu_time();
	clearStructure(nearestSite, image.rows * image.cols);
	clearStructure(proximateSites, image.rows * image.cols);
	computeNearestSiteInRow(image, nearestSite);
	computeProximateSitesInColumn(nearestSite, proximateSites, image.rows, image.cols);
	computeNearestSiteInFull(proximateSites, nearestSite, image.rows, image.cols);
	computeDistanceTransform(EDTImage, nearestSite, image.rows, image.cols);
	double end = cpu_time();
	printf("%f ms\n", (end - begin) * 1000);
	
	while(cv::waitKey(33) != 13) {
		cv::imshow("Original Image", image);
		cv::imshow("EDT Image", EDTImage);
	}

	delete [] nearestSite;
	delete [] proximateSites;
	return 0;

}