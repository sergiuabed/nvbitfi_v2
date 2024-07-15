#include "utils.h"

void init_mat_device(float* mat, int rsize, int csize, int seed){
    srand(seed);

    for(int i=0; i < rsize; i++){
        for(int j=0; j < csize; j++){
            mat[i*csize + j] = ((float) rand()) / ((float) RAND_MAX);
        }
    }
}

float** init_mat_host(int rsize, int csize, int seed){

    //malloc matrix
    float** m = (float**) malloc(rsize * sizeof(float*));
    if(m == NULL){
        fprintf(stderr, "Error allocating array of floats*\n");
        exit(1);
    }

    for(int i=0; i<rsize; i++){
        m[i] = (float*) malloc(csize * sizeof(float));
        if(m[i] == NULL){
            fprintf(stderr, "Error allocating arrays of floats\n");
            exit(1);
        }
    }

    srand(seed);

    for(int i=0; i < rsize; i++){
        for(int j=0; j < csize; j++){
            m[i][j] = ((float) rand()) / ((float) RAND_MAX);
        }
    }

    return m;
}

float** matmul_host(float* A, float* B, int m, int n, int k){
    //malloc matrix
    float** C = (float**) malloc(m * sizeof(float*));
    if(C == NULL){
        fprintf(stderr, "Error allocating array of floats*\n");
        exit(1);
    }

    for(int i=0; i<m; i++){
        C[i] = (float*) malloc(k * sizeof(float));
        if(C[i] == NULL){
            fprintf(stderr, "Error allocating arrays of floats\n");
            exit(1);
        }
    }

    // matrix multiplication
    for(int i = 0; i < m; i++){
        for(int j=0; j < k; j++){
            C[i][j] = 0;

            for(int l=0; l < n; l++){
                C[i][j] += A[i*n + l] * B[l*n + j];
            }
        }
    }

    return C;
}

bool check_correctness(float* Cdevice, float** Chost, int rows, int cols){

    for(int i=10; i < rows; i++){
        for(int j=10; j < cols; j++)
            if(Cdevice[i*cols + j] - Chost[i][j] > 1e-2){
                std::cout << "WRONG!!" << std::endl;
                std::cout << "At (i = " << i << ", j = " << j << ") we have:" << std::endl;
                std::cout << "device: " << Cdevice[i*cols + j] << std::endl;
                std::cout << "host: " << Chost[i][j] << std::endl;
                return false;
            }
    }

    return true;
}

float sum_of_entries(float* m, int rows, int cols){
    float sum = 0;
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++){
            sum += m[i*cols + j];
        }
    return sum;
}

float get_max(float* m, int rows, int cols){
    float max = m[0];
    for(int i = 1; i < rows*cols; i++){
        if(m[i] > max)
            max = m[i];
    }

    return max;
}