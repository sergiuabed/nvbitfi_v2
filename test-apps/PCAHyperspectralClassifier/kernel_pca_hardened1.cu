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

int host_check_equality(float* T, float*T2, int size)
{
	for(int i=0; i < size; i++)
	{
		if(T[i] != T2[i])
		{
			std::cout << "Matrices are different at position " << i << std::endl;
			return 0; // return False
		}
	}

	std::cout << "Matrices are equal!" << std::endl;
	return 1; // return True
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

	// transfer the host matrix R to device matrix dR2 (redundancy)
	float *dR2 = 0;
	status = cublasAlloc(M*N, sizeof(dR2[0]), (void**)&dR2);

	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dR2)\n");
	}

	status = cublasSetMatrix(M, N, sizeof(R[0]), R, M, dR2, M);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda access error (write dR2)\n");
	}

	// transfer the host matrix R to device matrix dR3 (2nd redundancy)
	float *dR3 = 0;
	status = cublasAlloc(M*N, sizeof(dR3[0]), (void**)&dR3);

	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dR3)\n");
	}

	status = cublasSetMatrix(M, N, sizeof(R[0]), R, M, dR3, M);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda access error (write dR3)\n");
	}

	// allocate device memory for T, P
	float *dT = 0;
	status = cublasAlloc(M*K, sizeof(dT[0]), (void**)&dT);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dT)\n");
	}

	// allocate device memory for redundant copy of T for hardening
	float *dT2 = 0;
	status = cublasAlloc(M*K, sizeof(dT2[0]), (void**)&dT2);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dT2)\n");
	}

	float *dT2_copy = 0;
	status = cublasAlloc(M*K, sizeof(dT2_copy[0]), (void**)&dT2_copy);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dT2_copy)\n");
	}

	float *dT3 = 0;
	status = cublasAlloc(M*K, sizeof(dT3[0]), (void**)&dT3);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dT3)\n");
	}

	float *dP = 0;
	status = cublasAlloc(N*K, sizeof(dP[0]), (void**)&dP);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dP)\n");
	}

	float *dP2 = 0;
	status = cublasAlloc(N*K, sizeof(dP2[0]), (void**)&dP2);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dP2)\n");
	}

	float *dP3 = 0;
	status = cublasAlloc(N*K, sizeof(dP3[0]), (void**)&dP3);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dP3)\n");
	}

	// allocate memory for eigenvalues
	float *L;
	L = (float*)malloc(K * sizeof(L[0]));;
	if(L == 0)
	{
		std::runtime_error( "! memory allocation error: T\n");
	}

	// allocate memory for eigenvalues (redundancy)
	float *L2;
	L2 = (float*)malloc(K * sizeof(L2[0]));;
	if(L2 == 0)
	{
		std::runtime_error( "! memory allocation error: T\n");
	}

	// allocate memory for eigenvalues (redundancy)
	float *L3;
	L3 = (float*)malloc(K * sizeof(L3[0]));;
	if(L3 == 0)
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

	// mean center the data (redundancy)
	float *dU2 = 0;
	status = cublasAlloc(M, sizeof(dU2[0]), (void**)&dU2);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dU2)\n");
	}

	// mean center the data (2nd redundancy)
	float *dU3 = 0;
	status = cublasAlloc(M, sizeof(dU3[0]), (void**)&dU3);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		std::runtime_error( "! cuda memory allocation error (dU3)\n");
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
	
	//redundancy
	cublasScopy(M, &dR2[0], 1, dU2, 1);
	for(n=1; n<N; n++)
	{
		cublasSaxpy (M, 1.0, &dR2[n*M], 1, dU2, 1);
	}

	for(n=0; n<N; n++)
	{
		cublasSaxpy (M, -1.0/N, dU2, 1, &dR2[n*M], 1);
	}

	//2nd redundancy
	cublasScopy(M, &dR3[0], 1, dU3, 1);
	for(n=1; n<N; n++)
	{
		cublasSaxpy (M, 1.0, &dR3[n*M], 1, dU3, 1);
	}

	for(n=0; n<N; n++)
	{
		cublasSaxpy (M, -1.0/N, dU3, 1, &dR3[n*M], 1);
	}

	// GS-PCA
	int isEqual=1;
	float a;
	float a2;
	float a3;
	for(k=0; k<K; k++)
	{
		cublasScopy (M, &dR[k*M], 1, &dT[k*M], 1);

		//redundancy
		cublasScopy (M, &dR2[k*M], 1, &dT2[k*M], 1);

//		//*******CHECK CORRECTNESS*********
//		cublasScopy(M, &dT2[k*M], 1, &dT3[k*M], 1);
//		cublasSaxpy(N*k, -1.0, dT, 1, dT3, 1);
//
//		isEqual = cublasSasum(N*k, dT3, 1);
//
//		if(isEqual != 0)
//		{
//			std::cout << "dT AND dT2 ARE DIFFERENT!!" << std::endl;
//		}
//		else
//		{
//			std::cout << "dT and dT2 are equal" << std::endl;
//		}
//
//		//*********************************



		a = 0.0;
		a2 = 0.0;
		a3 = 0.0;
		for(j=0; j<J; j++)
		{
			//**********SAVE VARIABLES COPIES BEFORE CHANGING THEM**********

			cublasScopy(M*N, dR, 1, dR3, 1);
			cublasScopy(k*M, dT, 1, dT3, 1);
			cublasScopy(N*K, dP, 1, dP3, 1);
			L3[k] = L[k];
			cublasScopy(M, dU, 1, dU3, 1);

			//**************************************************************

			cublasSgemv ('t', M, N, 1.0, dR, M, &dT[k*M], 1, 0.0, &dP[k*N], 1);

			//redundancy
			cublasSgemv ('t', M, N, 1.0, dR2, M, &dT2[k*M], 1, 0.0, &dP2[k*N], 1);

			

			if(k>0)
			{
				cublasSgemv ('t', N, k, 1.0, dP, N, &dP[k*N], 1, 0.0, dU, 1);
				cublasSgemv ('n', N, k, -1.0, dP, N, dU, 1, 1.0, &dP[k*N], 1);

				//redundancy
				cublasSgemv ('t', N, k, 1.0, dP2, N, &dP2[k*N], 1, 0.0, dU2, 1);
				cublasSgemv ('n', N, k, -1.0, dP2, N, dU2, 1, 1.0, &dP2[k*N], 1);
			}
			cublasSscal (N, 1.0/cublasSnrm2(N, &dP[k*N], 1), &dP[k*N], 1);
			cublasSgemv ('n', M, N, 1.0, dR, M, &dP[k*N], 1, 0.0, &dT[k*M], 1);

			//redundancy
			cublasSscal (N, 1.0/cublasSnrm2(N, &dP2[k*N], 1), &dP2[k*N], 1);
			cublasSgemv ('n', M, N, 1.0, dR2, M, &dP2[k*N], 1, 0.0, &dT2[k*M], 1);

			if(k>0)
			{
				cublasSgemv ('t', M, k, 1.0, dT, M, &dT[k*M], 1, 0.0, dU, 1);
				cublasSgemv ('n', M, k, -1.0, dT, M, dU, 1, 1.0, &dT[k*M], 1);

				//redundancy
				cublasSgemv ('t', M, k, 1.0, dT2, M, &dT2[k*M], 1, 0.0, dU2, 1);
				cublasSgemv ('n', M, k, -1.0, dT2, M, dU2, 1, 1.0, &dT2[k*M], 1);

			}

//			float *T = (float*)malloc(sizeof(float)*M*K);
//			float *T2 = (float*)malloc(sizeof(float)*M*K);
//
//			cudaMemcpy(T, dT, sizeof(dT[0])*M*K, cudaMemcpyDeviceToHost);
//			cudaMemcpy(T2, dT2, sizeof(dT2[0])*M*K, cudaMemcpyDeviceToHost);

			

			L[k] = cublasSnrm2(M, &dT[k*M], 1);
			cublasSscal(M, 1.0/L[k], &dT[k*M], 1);

			// redundancy
			L2[k] = cublasSnrm2(M, &dT2[k*M], 1);
			cublasSscal(M, 1.0/L2[k], &dT2[k*M], 1);

			if(fabs(a - L[k]) < er*L[k]) break;
			if(fabs(a2 - L2[k]) < er*L2[k]) break;
			
			a = L[k];
			a2 = L2[k];
			
			//********************
				//HERE WE DO MATRICES EQUALITY CHECK

			//compute difference between matrix dT and its redundant copy dT2
			cublasScopy(M*k, dT2, 1, dT2_copy, 1);
			cublasSaxpy(M*k, -1.0, dT, 1, dT2_copy, 1);

			//sum the elements within dT2_copy which now stores the result of the previous operation
			isEqual = 0;
			isEqual = cublasSasum(k*M, dT2_copy, 1);

			if(isEqual != 0)
			{
				//std::cout << "Not equal!! isEqual = " << isEqual << "; j=" << j << "k=" << k << std::endl;
				//std::cout << "Not equal!! isEqual = " << isEqual << "; dT2[0]" << dT2[0] << std::endl;
				//std::cout << "Size of dT2: " << sizeof(*dT2) << std::endl;

				//****************** NEED TO EXECUTE A 3RD TIME *****************************
				
				//dR = dR3;
				//dT = dT3;
				//dP = dP3;
				//L[k] = L3[k];
				//dU = dU3;

				cublasSgemv ('t', M, N, 1.0, dR3, M, &dT3[k*M], 1, 0.0, &dP3[k*N], 1);
				if(k>0)
				{
					cublasSgemv ('t', N, k, 1.0, dP3, N, &dP3[k*N], 1, 0.0, dU3, 1);
					cublasSgemv ('n', N, k, -1.0, dP3, N, dU3, 1, 1.0, &dP3[k*N], 1);
				}
				cublasSscal (N, 1.0/cublasSnrm2(N, &dP3[k*N], 1), &dP3[k*N], 1);
				cublasSgemv ('n', M, N, 1.0, dR3, M, &dP3[k*N], 1, 0.0, &dT3[k*M], 1);
				if(k>0)
				{
					cublasSgemv ('t', M, k, 1.0, dT3, M, &dT3[k*M], 1, 0.0, dU3, 1);
					cublasSgemv ('n', M, k, -1.0, dT3, M, dU3, 1, 1.0, &dT3[k*M], 1);
				}

				L3[k] = cublasSnrm2(M, &dT3[k*M], 1);
				cublasSscal(M, 1.0/L3[k], &dT3[k*M], 1);

				if(fabs(a3 - L3[k]) < er*L3[k]) break;

				a3 = L3[k];

				// copy the '3' versions into the '1's and '2's
				cublasScopy(M*N, dR3, 1, dR, 1);
				cublasScopy(k*M, dT3, 1, dT, 1);
				cublasScopy(N*K, dP3, 1, dP, 1);
				L[k] = L3[k];
				cublasScopy(M, dU3, 1, dU, 1);

				cublasScopy(M*N, dR3, 1, dR2, 1);
				cublasScopy(k*M, dT3, 1, dT2, 1);
				cublasScopy(N*K, dP3, 1, dP2, 1);
				L2[k] = L3[k];
				cublasScopy(M, dU3, 1, dU2, 1);
			}
			//********************

		}
			
		cublasSger (M, N, - L[k], &dT[k*M], 1, &dP[k*N], 1, dR, M);

		// redundancy
		cublasSger (M, N, - L2[k], &dT2[k*M], 1, &dP2[k*N], 1, dR2, M);
	
			
	
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

	status = cublasFree(dP2);
	status = cublasFree(dT2);
	status = cublasFree(dT2_copy);
	status = cublasFree(dR2);
	status = cublasFree(dU2);

	status = cublasFree(dP3);
	status = cublasFree(dT3);
	status = cublasFree(dR3);
	status = cublasFree(dU3);
}


void KernelPCA::set_n_components(int K_)
{
	K = K_;
}


int KernelPCA::get_n_components()
{
	return K;
}



