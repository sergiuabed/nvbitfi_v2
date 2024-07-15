#include "kernels.cuh"

__global__ void MatrixMulKernel(float* A, float* B, float* C, int m, int n, int k, int tile_width){
    /*
    DYNAMIC ALLOCATION IN SHARED MEMORY:
    When executing the kernel function from host code, we use the operator 
    <<<numBlocks, numThreadsPerBlock, sharedMem, stream>>>
    
    To dynamically allocate memory in the shared memory, we just do
    "extern __shared__ DATATYPE var_name[]", i.e. we add extern to the
    variable declaration. This indicates that the memory has been allocated
    somewhere else (most likely done by some CUDA routine) and we just use
    extern to access it.

    So, we don't have to explicitly allocate it. CUDA takes care of it.

    Also, we don't need to explicitly free the dynamically allocated shared memory,
    since it is automatically done at the end of the kernel execution
    */
    
    // A (m x n) and B (n x k) are the matrices to be multiplied
    // C (m x k) is the result matrix

    // dynamic allocation of A and B
    extern __shared__ float shared_mem[];

    float* dA = (float*) (shared_mem);
    float* dB = (float*) (shared_mem + tile_width * tile_width);
    //float* dC = (float*) (shared_mem + 2 * tile_width * tile_width);

    // define Cvalue corresponding to the thread
    float Cvalue = 0.0; // register variable

    // get global thread idx
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for(int t = 0; t < (n - 1)/tile_width + 1; t++){
        // check boundary conditions
        if(row < m && (tile_width*t + tx) < n)
            dA[ty*tile_width + tx] = A[row * n + t * tile_width + tx];
        else
            dA[ty*tile_width + tx] = 0.0;
        
        if(t * tile_width + ty < n && col < k)
            dB[ty*tile_width + tx] = B[(t * tile_width + ty) * k + col]; // B[t * tile_width * k + ty * k + col]
        else
            dB[ty*tile_width + tx] = 0.0;

        __syncthreads();

        for(int i = 0; i < tile_width; i++)
            Cvalue += dA[ty*tile_width + i] * dB[i*tile_width + tx];
        
        __syncthreads();
    }

    if(row < m && col < k){
        C[row*k + col] = Cvalue;
    }

}


int main(){
    cudaError_t err = cudaSuccess;

    //allocate pageable memory
    float* h_a = (float*) malloc(M * N * sizeof(float));
    float* h_b = (float*) malloc(N * K * sizeof(float));
    float* h_c = (float*) malloc(M * K * sizeof(float));

    //init matrices
    init_mat_device(h_a, M, N, 1);
    init_mat_device(h_b, N, K, 2);

    std::cout << h_a[0] << h_a[2] << h_a[10] << std::endl;
    std::cout << h_b[0] << h_b[2] << h_b[10] << std::endl;


    //allocate device memory
    float* d_a, *d_b, *d_c;

    cudaMalloc((float**)&d_a, M * N * sizeof(float));
    cudaMemcpy(d_a, h_a, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((float**)&d_b, N * K * sizeof(float));
    cudaMemcpy(d_b, h_b, N * K * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((float**)&d_c, M * K * sizeof(float));

    //setup grid and blocks configuration
    int gridRows = (M - 1)/TILE_WIDTH + 1;
    int gridCols = (K - 1)/TILE_WIDTH + 1;
    dim3 blocksInGrid(gridRows, gridCols);// CHECK WHETHER TO SWAP gridRows and gridCols
    //dim3 blocksInGrid(gridCols, gridRows);
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    
    int shared_mem_size = 2 * TILE_WIDTH * TILE_WIDTH;//int shared_mem_size = (N*M + M*K + N*K) * sizeof(float);

    //execute kernel
    MatrixMulKernel<<<blocksInGrid, threadsPerBlock, shared_mem_size>>>(d_a, d_b, d_c, M, N, K, TILE_WIDTH);
    //MatrixMulKernel_static<<<blocksInGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K, TILE_WIDTH);


    //copy results from device to memory
    err = cudaMemcpy(h_c, d_c, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    if(err != cudaSuccess){
        std::cerr << "WHY?????" << std::endl;
    }

    std::cout << "Sums of C entries from device: " << sum_of_entries(h_c, M, K) << std::endl;
    std::cout << h_c[0] << h_c[2] << h_c[10] << h_c[50] << std::endl;

    //--------------------------

    // compare device result with host result
    float** hC = matmul_host(h_a, h_b, M, N, K);

    std::cout << "something" << std::endl;
    std::cout << hC[0][1] << hC[2][2] << hC[10][3] << std::endl;


    bool flag = check_correctness(h_c, hC, M, K);

    std::cout << "flag is " << flag << std::endl;
    if(flag)
        std::cout << "Matrix multiplication correct!" << std::endl;
    else
        std::cout << "Something went wrong!" << std::endl;

    std::cout << "Matrix A max: " << get_max(h_a, M, N) << std::endl;
    std::cout << "Matrix B max: " << get_max(h_b, N, K) << std::endl;
    std::cout << "Matrix C max: " << get_max(h_c, M, K) << std::endl;

    //free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(hC);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}