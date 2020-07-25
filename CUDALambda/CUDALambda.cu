#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <vector>

#include "helper.h"


/**
 * kernel caller for the lambda function
 */
template <typename Func>
__global__ void lambda_caller(Func f, const int size, int* x)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int my_x = x[idx];
        x[idx] = f(my_x);
    }
}

/**
 * Functor struct
 */
struct Functor
{
    __device__ int operator()(int x)
    {
        return x * x;
    }
};

/**
 * kernel caller for the functor
 */
template <typename Func>
__global__ void functor_caller(Func f, const int size, int* x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int my_x = x[idx];
        x[idx] = (*f)(my_x);
    }
}


/**
 * Host verifier
 */
bool verify(const int* d_x, const int size, const int truth)
{
    std::vector<int> h_x(size);

    CUDA_ERROR(cudaMemcpy(h_x.data(), d_x, size * sizeof(int),
                          cudaMemcpyDeviceToHost));

    for (const int v : h_x) {
        if (v != truth) {
            return false;
        }
    }
    return true;
}

float test_lambda(const int size,
                  const int val,
                  const int num_threads,
                  const int num_blocks)
{
    const int        bytes = size * sizeof(int);
    std::vector<int> h_x(size, val);
    int*             d_x(nullptr);

    CUDA_ERROR(cudaMalloc((void**)&d_x, bytes));
    CUDA_ERROR(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));


    auto transform_lambda = [=] __host__ __device__(int x) { return x * x; };

    CUDATimer timer;
    timer.start();

    lambda_caller<<<num_blocks, num_threads>>>(transform_lambda, size, d_x);
    CUDA_ERROR(cudaDeviceSynchronize());

    timer.stop();

    if (verify(d_x, size, val * val)) {
        //printf("\nLambda test passed\n");
    } else {
        printf("\nLambda test FAILED!!!!\n");
    }
    CUDA_ERROR(cudaFree(d_x));

    return timer.elapsed_millis();
}

float test_functor(const int size,
                   const int val,
                   const int num_threads,
                   const int num_blocks)
{
    const int        bytes = size * sizeof(int);
    std::vector<int> h_x(size, val);
    int*             d_x(nullptr);

    CUDA_ERROR(cudaMalloc((void**)&d_x, bytes));
    CUDA_ERROR(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    Functor *d_transform_functor(nullptr), h_transform_functor;
    CUDA_ERROR(cudaMalloc((void**)&d_transform_functor, sizeof(Functor)));
    CUDA_ERROR(cudaMemcpy(d_transform_functor, &h_transform_functor,
                          sizeof(Functor), cudaMemcpyHostToDevice));

    CUDATimer timer;
    timer.start();

    functor_caller<<<num_blocks, num_threads>>>(d_transform_functor, size, d_x);
    CUDA_ERROR(cudaDeviceSynchronize());

    timer.stop();

    if (verify(d_x, size, val * val)) {
       // printf("\nFunctor test passed\n");
    } else {
        printf("\nFunctor test FAILED!!!!\n");
    }


    CUDA_ERROR(cudaFree(d_transform_functor));
    CUDA_ERROR(cudaFree(d_x));

    return timer.elapsed_millis();
}


int main(int argc, char** argv)
{
    const int num_run = 10;
    const int size = 1024*1024*10;
    const int val = 10;
    const int num_threads = 512;
    const int num_blocks = DIVIDE_UP(size, num_threads);

    printf("\n size = %d, num_run = %d", size, num_run);

    {
        float functor_time = 0;
        for (int i = 0; i < num_run; ++i) {
            functor_time += test_functor(size, val, num_threads, num_blocks);
        }
        printf("\n average test_functor time = %f", functor_time / num_run);
    }

     {
        float lambda_time = 0;
        for (int i = 0; i < num_run; ++i) {
            lambda_time += test_lambda(size, val, num_threads, num_blocks);
        }
        printf("\n average test_lambda time = %f", lambda_time / num_run);
    }
   

    return EXIT_SUCCESS;
}