/***
This software preprocess data using Principal Component Analysis ( PCA ) exploiting CUDA.
Modified by Gianluca De Lucia ( gianluca.delucia.94@gmail.com ) and Diego Romano ( diego.romano@cnr.it )
based on GPU_GSPCA code by Nathaniel Merrill.
***/

#include "kernel_pca.h"
#include <iostream>


KernelPCA::KernelPCA() : K(-1)
{
        // initialize cublas
        status = cublasInit();

        if(status != CUBLAS_STATUS_SUCCESS)
        {
                std::runtime_error( "! CUBLAS initialization error\n");
        }
}



KernelPCA::KernelPCA(int num_pcs) : K(num_pcs)
{
        // initialize cublas
        status = cublasInit();

        if(status != CUBLAS_STATUS_SUCCESS)
        {
                std::runtime_error( "! CUBLAS initialization error\n");
        }
}




KernelPCA::~KernelPCA()
{
	
        // shutdown
        status = cublasShutdown(); 
        if(status != CUBLAS_STATUS_SUCCESS) 
        { 
                std::runtime_error( "! cublas shutdown error\n"); 
        } 


}

void KernelPCA::fit_transform(int M, int N, float *R, bool verbose, float* imgT)
{


	// maximum number of iterations
	int J = 10000;

	// max error
	float er = 1.0e-7;

        // if no K specified, or K > min(M, N)
        int K_;
        K_ = min(M, N);
        if (K == -1 || K > K_) K = K_;

	int n, j, k;

	// transfer the host matrix R to device matrix dR
	float *dR = 0;
	status = cublasAlloc(M*N, sizeof(dR[0]), (void**)&dR);

	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dR)\n");
	}

	status = cublasSetMatrix(M, N, sizeof(R[0]), R, M, dR, M);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda access error (write dR)\n");
	}

	// allocate device memory for T, P
	float *dT = 0;
	status = cublasAlloc(M*K, sizeof(dT[0]), (void**)&dT);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dT)\n");
	}

	float *dP = 0;
	status = cublasAlloc(N*K, sizeof(dP[0]), (void**)&dP);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dP)\n");
	}

	// allocate memory for eigenvalues
	float *L;
	L = (float*)malloc(K * sizeof(L[0]));;
	if(L == 0)
	{
		std::runtime_error( "! memory allocation error: T\n");
	}

	// mean center the data
	float *dU = 0;
	status = cublasAlloc(M, sizeof(dU[0]), (void**)&dU);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dU)\n");
	}

	cublasScopy(M, &dR[0], 1, dU, 1);
	for(n=1; n<N; n++)
	{
		cublasSaxpy (M, 1.0, &dR[n*M], 1, dU, 1);
	}

	for(n=0; n<N; n++)
	{
		cublasSaxpy (M, -1.0/N, dU, 1, &dR[n*M], 1);
	}
	
	// GS-PCA
	float a;
	for(k=0; k<K; k++)
	{
		cublasScopy (M, &dR[k*M], 1, &dT[k*M], 1);
		a = 0.0;
		for(j=0; j<J; j++)
		{
			cublasSgemv ('t', M, N, 1.0, dR, M, &dT[k*M], 1, 0.0, &dP[k*N], 1);
			if(k>0)
			{
				cublasSgemv ('t', N, k, 1.0, dP, N, &dP[k*N], 1, 0.0, dU, 1);
				cublasSgemv ('n', N, k, -1.0, dP, N, dU, 1, 1.0, &dP[k*N], 1);
			}
			cublasSscal (N, 1.0/cublasSnrm2(N, &dP[k*N], 1), &dP[k*N], 1);
			cublasSgemv ('n', M, N, 1.0, dR, M, &dP[k*N], 1, 0.0, &dT[k*M], 1);
			if(k>0)
			{
				cublasSgemv ('t', M, k, 1.0, dT, M, &dT[k*M], 1, 0.0, dU, 1);
				cublasSgemv ('n', M, k, -1.0, dT, M, dU, 1, 1.0, &dT[k*M], 1);
			}

			L[k] = cublasSnrm2(M, &dT[k*M], 1);
			cublasSscal(M, 1.0/L[k], &dT[k*M], 1);

			if(fabs(a - L[k]) < er*L[k]) break;
			
			a = L[k];
			
		}
			
		cublasSger (M, N, - L[k], &dT[k*M], 1, &dP[k*N], 1, dR, M);
	
			
	
	}


	for(k=0; k<K; k++)
	{
		cublasSscal(M, L[k], &dT[k*M], 1);
	}

	float *T;
        T = (float*)malloc(M*K * sizeof(T[0])); // user needs to free this outside this function

        if(T == 0)
        {
                std::runtime_error("! memory allocation error: T\n");
        }


	// transfer device dT to host T
	//cublasGetMatrix (M, K, sizeof(dT[0]), dT, M, imgT, M);
        cudaMemcpy(imgT,dT,sizeof(dT[0])*M*K,cudaMemcpyDeviceToDevice);


	// clean up memory
	free(L);
	status = cublasFree(dP);
	status = cublasFree(dT);
	status = cublasFree(dR);
	status = cublasFree(dU);

}


void KernelPCA::set_n_components(int K_)
{
	K = K_;
}


int KernelPCA::get_n_components()
{
	return K;
}



