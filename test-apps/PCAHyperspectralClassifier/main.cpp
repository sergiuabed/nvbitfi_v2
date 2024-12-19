/***

This software preprocess data using Principal Component Analysis ( PCA ) exploiting CUDA.
Developed by Gianluca De Lucia ( gianluca.delucia.94@gmail.com ) and Diego Romano ( diego.romano@cnr.it )

***/

#include "kernel_pca.h"
#include <string>
#include <iostream>
#include <iomanip>


using namespace std;
void mainFunction(float* img, int K, int d0, int d1, int d2, float* imgT);
void mainFunction_hardened(float* img, int K, int d0, int d1, int d2, float* imgT);
bool check_equality(float* imgT1, float* imgT2, int size);

void mainFunction(float* img, int K, int d0, int d1, int d2, float* imgT){
		
		int M, N, m, n;

		// initialize srand and clock
	    srand (time(NULL));

		//from cube 3D to matrix 2D		
		M = d0*d1;
	    N = d2;
		double dtime;
		clock_t start;
		KernelPCA* pca;
		//float *T = (float*)malloc(sizeof(float)*d0*d1*K);
		float *T = (float*)malloc(sizeof(float)*d0*d1*K);
		float *T0 = (float*)malloc(sizeof(float)*d0*d1*d2);

		for (int i = 0; i < d0*d1*d2 ; ++i){
			T0[i] = img[i];
		}


		pca = new KernelPCA(K);

	    start=clock();

	    pca->fit_transform(M, N, img, 1,imgT);

	    dtime = ((double)clock()-start)/CLOCKS_PER_SEC;

		printf("\nTime for GS-PCA in CUBLAS: %f seconds\n", dtime);

	}

void mainFunction_hardened(float* img, int K, int d0, int d1, int d2, float* imgT){
		
		int M, N, m, n;

		// initialize srand and clock
	    srand (time(NULL));

		//from cube 3D to matrix 2D		
		M = d0*d1;
	    N = d2;
		double dtime;
		clock_t start;
		KernelPCA* pca;
		KernelPCA* pca1;
		KernelPCA* pca2;
		//float *T = (float*)malloc(sizeof(float)*d0*d1*K);
		float *imgT_host1 = (float*)malloc(sizeof(float)*d0*d1*K);
		float *imgT_host2 = (float*)malloc(sizeof(float)*d0*d1*K);
		
		if(imgT_host1 == NULL){
			cerr << "Error while allocating memory!" << endl;
			exit(0);
		}

		if(imgT_host2 == NULL){
			cerr << "Error while allocating memory!" << endl;
			exit(0);
		}

		float *T0 = (float*)malloc(sizeof(float)*d0*d1*d2);

		for (int i = 0; i < d0*d1*d2 ; ++i){
			T0[i] = img[i];
		}

		// Hardened block
		pca1 = new KernelPCA(K);
		pca2 = new KernelPCA(K);

	    start=clock();

	    pca1->fit_transform(M, N, img, 1,imgT);
		cudaMemcpy(imgT_host1, imgT, sizeof(imgT_host1[0])*M*K, cudaMemcpyDeviceToHost);

		pca2->fit_transform(M, N, img, 1,imgT);
		cudaMemcpy(imgT_host2, imgT, sizeof(imgT_host2[0])*M*K, cudaMemcpyDeviceToHost);

		// check equality
		if(!check_equality(imgT_host1, imgT_host2, d0*d1*K))
		{
			pca = new KernelPCA(K);
			pca->fit_transform(M, N, img, 1,imgT);
		}

		// end hardening

	    dtime = ((double)clock()-start)/CLOCKS_PER_SEC;

		printf("\nTime for GS-PCA in CUBLAS: %f seconds\n", dtime);

	}

bool check_equality(float* imgT1, float* imgT2, int size)
{
	for(int i = 0; i < size; i++)
		if(imgT1[i] != imgT2[i])
		{
			//cout << "DwC: not equal! Different at index "<< i << endl;
			return false;
		}

		//cout << "I'm writing " << imgT1[i] << endl;
	return true;
}

extern "C" {
    void cudaPCA(float* img,int K, int d0, int d1, int d2, float* imgT)
    {
        //return mainFunction(img,K,d0,d1,d2, imgT);
		return mainFunction_hardened(img,K,d0,d1,d2, imgT);
    }
}



