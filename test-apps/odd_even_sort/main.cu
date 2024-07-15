#include <iostream>

#define N 10

__device__ bool check_sorted(float* a, int length){
    for(int i = 0; i < length - 1; i++){
        if(a[i] > a[i + 1])
            return false;
    }

    return true;
}

__global__ void OddEven_kernel(float* a, int length){
    int pos = 2 * threadIdx.x;
    int unit = 1;
    float aux;

    while(!check_sorted(a, length)){
        if(pos < length - 1){
            if(a[pos] > a[pos + 1]){
                aux = a[pos];
                a[pos] = a[pos + 1];
                a[pos + 1] = aux;
            }
        }
        __syncthreads();
        pos += unit;
        unit = -unit;
    }

}

void init_arr(float* a, int length){
    srand(42);

    for(int i = 0; i < length; i++){
        a[i] = ((float) rand()) / ((float) RAND_MAX);
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

int main(){
    cudaError_t err = cudaSuccess;

    //allocate pageable memory
    float* arr = (float*) malloc(N * sizeof(float));

    if(arr == NULL){
        std::cerr << "Error allocating array in host memory" << std::endl;
        exit(1);
    }

    //init array
    init_arr(arr, N);

    //allocate device memory
    float *d_arr;

    err = cudaMalloc((float**)&d_arr, N*sizeof(float));
    if(err != cudaSuccess){
        std::cerr << "Error allocating array in host memory" << std::endl;
        exit(1);
    }

    err = cudaMemcpy(d_arr, arr, N*sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        std::cerr << "Error copying from host to device" << std::endl;
        exit(1);
    }

    //kernel launch
    std::cout << "I'm here" << std::endl;
    OddEven_kernel<<<1, N/2>>>(d_arr, N);
    std::cout << "I'm out" << std::endl;

    err = cudaMemcpy(arr, d_arr, N*sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        std::cerr << "Error copying from host to device" << std::endl;
        exit(1);
    }

    //print result
    for(int i=0; i < N; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;

}