#ifndef UTILS
#define UTILS

#include <iostream>
#include <string>

#define M 98
#define N 100
#define K 150

#define TILE_WIDTH 32

void init_mat_device(float*, int, int, int);
float** init_mat_host(int, int, int);
float** matmul_host(float*, float*, int, int, int);
bool check_correctness(float*, float**, int, int);
float sum_of_entries(float*, int, int);
float get_max(float*, int, int);

#endif