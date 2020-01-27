#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "unmaxpool_layer.h"
#include "cuda.h"
}

__global__ void forward_unmaxpool_layer_kernel(int n, float *input, float *output, int *indexes)
{

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int index_id = indexes[id];
    float val = input[id];
    output[index_id] = val;
}

__global__ void backward_unmaxpool_layer_kernel(int n, float *delta, float *prev_delta, int *indexes)
{

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int index_id = indexes[id];
    float val = delta[index_id];
    prev_delta[id] = val;
}

extern "C" void forward_unmaxpool_layer_gpu(unmaxpool_layer layer, network_state state)
{
    int h = layer.h;
    int w = layer.w;
    int c = layer.c;

    int layer_id = layer.pointLayer;
    float * layer_index = state.net.layers[layer_id].indexes_gpu;

    size_t n = h*w*c*layer.batch;

    forward_unmaxpool_layer_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream()>>>(n, state.input, layer.output_gpu, layer_index);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_unmaxpool_layer_gpu(unmaxpool_layer layer, network_state state)
{
    size_t n = layer.h*layer.w*layer.c*layer.batch;
    int layer_id = layer.pointLayer;
    float * layer_index = state.net.layers[layer_id].indexes_gpu;

    backward_unmaxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.delta_gpu, state.delta, layer_index);
    check_error(cudaPeekAtLastError());
}

