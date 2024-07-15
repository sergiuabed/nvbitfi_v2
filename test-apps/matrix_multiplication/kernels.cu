#include "utils.h"

//__global__ void MatrixMulKernel(float* A, float* B, float* C, int m, int n, int k, int tile_width){
//    /*
//    DYNAMIC ALLOCATION IN SHARED MEMORY:
//    When executing the kernel function from host code, we use the operator 
//    <<<numBlocks, numThreadsPerBlock, sharedMem, stream>>>
//    
//    To dynamically allocate memory in the shared memory, we just do
//    "extern __shared__ DATATYPE var_name[]", i.e. we add extern to the
//    variable declaration. This indicates that the memory has been allocated
//    somewhere else (most likely done by some CUDA routine) and we just use
//    extern to access it.
//
//    So, we don't have to explicitly allocate it. CUDA takes care of it.
//
//    Also, we don't need to explicitly free the dynamically allocated shared memory,
//    since it is automatically done at the end of the kernel execution
//    */
//    
//    // A (m x n) and B (n x k) are the matrices to be multiplied
//    // C (m x k) is the result matrix
//
//    // dynamic allocation of A and B
//    extern __shared__ float shared_mem[];
//
//    float* dA = (float*) (shared_mem);
//    float* dB = (float*) (shared_mem + tile_width * tile_width);
//    //float* dC = (float*) (shared_mem + 2 * tile_width * tile_width);
//
//    // define Cvalue corresponding to the thread
//    float Cvalue = 0.0; // register variable
//
//    // get global thread idx
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//    for(int t = 0; t < (n - 1)/tile_width + 1; t++){
//        // check boundary conditions
//        if(row < m && (tile_width*t + tx) < n)
//            dA[ty*tile_width + tx] = A[row * n + t * tile_width + tx];
//        else
//            dA[ty*tile_width + tx] = 0.0;
//        
//        if(t * tile_width + ty < n && col < k)
//            dB[ty*tile_width + tx] = B[(t * tile_width + ty) * k + col]; // B[t * tile_width * k + ty * k + col]
//        else
//            dB[ty*tile_width + tx] = 0.0;
//
//        __syncthreads();
//
//        for(int i = 0; i < tile_width; i++)
//            Cvalue += dA[ty*tile_width + i] * dB[i*tile_width + tx];
//        
//        __syncthreads();
//    }
//
//    if(row < m && col < k){
//        C[row*k + col] = Cvalue;
//    }
//
//}
//
//__global__ void MatrixMulKernel_static(float* A, float* B, float* C, int m, int n, int k, int tile_width){
//    // A (m x n) and B (n x k) are the matrices to be multiplied
//    // C (m x k) is the result matrix
//
//    // static allocation of A and B
//
//    __shared__ float dA[TILE_WIDTH][TILE_WIDTH];
//    __shared__ float dB[TILE_WIDTH][TILE_WIDTH];
//    //float* dC = (float*) (shared_mem + 2 * tile_width * tile_width);
//
//    // define Cvalue corresponding to the thread
//    float Cvalue = 0.0; // register variable
//
//    // get global thread idx
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//    
//    for(int t = 0; t < (n - 1)/tile_width + 1; t++){
//        //load tiles
//        // check boundary conditions
//        if(row < m && (tile_width*t + tx) < n){
//            dA[ty][tx] = A[row * n + t * tile_width + tx];
//        }
//        else{
//            dA[ty][tx] = 0.0;
//        }
//        
//        if((t * tile_width + ty) < n && col < k){
//            dB[ty][tx] = B[(t * tile_width + ty) * k + col]; // B[t * tile_width * k + ty * k + col]
//        }
//        else{
//            dB[ty][tx] = 0.0;
//        }
//
//        __syncthreads();
//
//
//        // compute partial results
//        for(int i = 0; i < tile_width; i++)
//            Cvalue += dA[ty][i] * dB[i][tx];
//        
//        __syncthreads();
//    }
//
//    if(row < m && col < k){
//        C[row*k + col] = Cvalue;
//    }
//
//}
