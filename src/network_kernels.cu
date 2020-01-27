#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "gru_layer.h"
#include "crnn_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"
}

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

void backward_only_network_gpu(network net)
{
    network_state state;
    state.index = 0;
    state.net = net;
    state.truth = 0;
    
    state.input = *net.input_gpu;
    state.delta = 0;
    state.train = 1;
#ifdef CUDNN_HALF
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		cuda_convert_f32_to_f16(l.weights_gpu, l.c*l.n*l.size*l.size, l.weights_gpu16);
	}
#endif
    backward_network_gpu(net, state);
	cudaStreamSynchronize(get_cuda_stream());
}

void forward_only_network_gpu(network net, float *x)
{
    network_state state;
    state.index = 0;
    state.net = net;
    state.truth = 0;
    int x_size = get_network_input_size(net)*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.train = 0;
#ifdef CUDNN_HALF
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		cuda_convert_f32_to_f16(l.weights_gpu, l.c*l.n*l.size*l.size, l.weights_gpu16);
	}
#endif
    forward_network_gpu(net, state);
	cudaStreamSynchronize(get_cuda_stream());
}

void forward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        //fprintf(stderr, "Layer %d \n",i);
        if(l.delta_gpu){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        l.forward_gpu(l, state);
        //fprintf(stderr, "CHECK forwards layer \n");
		if(net.wait_stream)
			cudaStreamSynchronize(get_cuda_stream());
        state.input = l.output_gpu;
    }
}

__global__ void update_sum_gpu(float* delta, float* delta_sum,
        const int height, const int width, const int channel, const int size, const int batch) {

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (index >= size)
        return;

    int col = (index % width);
    index = (index / width);
    int row = (index % height);
    index = (index / height);
    int b = (index % batch);

    int out_index = b*width*height + row*width + col;

    delta_sum[out_index] = 0;
    for (int c = 0; c < channel; c++){
        delta_sum[out_index] += abs(delta[b*channel*height*width + (col + width*(row + height*c))]);
    }
}

void update_sum_output(network net){

    layer last_l = net.layers[net.n-1];
    layer l = net.layers[net.n-1 + last_l.pointLayer];

    int w = l.w;
    int h = l.h;
    int batch = l.batch;

    int total = w*h*batch;

    float * delta = l.output_gpu;
    float * delta_sum = last_l.delta_sum_gpu;

    update_sum_gpu<<<cuda_gridsize(total), BLOCK>>>(delta, delta_sum, l.h, l.w, l.c, total, batch);

}

void update_sum(network net){

    layer last_l = net.layers[net.n-1];
    layer l = net.layers[net.n-1 + last_l.pointLayer];

    int w = l.w;
    int h = l.h;
    int batch = l.batch;

    int total = w*h*batch;

    float * delta = l.delta_gpu;
    float * delta_sum = last_l.delta_sum_gpu;

    update_sum_gpu<<<cuda_gridsize(total), BLOCK>>>(delta, delta_sum, l.h, l.w, l.c, total, batch);

}

__global__ void update_index_gpu(float* seen, float* delta_sum, float* part_index,
        const int height, const int width, const int batch, const int size, const int num_parts, const int total) {

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (index >= total)
        return;

    int col = (index % width);
    index = (index / width);
    int row = (index % height);
    index = (index / height);
    int b = (index % batch);

    int seen_offset = b * width * height * width * height + (row*width + col) * width * height;

    int max_i, max_j, max;
    int i,j,p;

    for(p = 0; p < num_parts; p++){
        max = -1;
        max_i = -1;
        max_j = -1;
        for(i = -size + row; i <= size + row; i++){
            for(j = -size + col; j <= size + col; j++){
                if((0 <= i < height) && (0 <= j < width)){
                    int out_index = b*width*height + i*width + j;
                    if((delta_sum[out_index] > max) && seen[seen_offset + max_i*width + max_j] < 1){
                        max = delta_sum[out_index];
                        max_i = i;
                        max_j = j;
                    }
                }
            }
        }
        int part_out_index = (b*width*height*num_parts + p*width*height + row*width + col) * 2;
        part_index[part_out_index + 0] = max_i;
        part_index[part_out_index + 1] = max_j;
        if(max_i > -1)
            seen[seen_offset + max_i*width + max_j] = 1;

    }
}

void update_all_index(network net){

    layer last_l = net.layers[net.n-1];
    layer l = net.layers[net.n-1 + last_l.pointLayer];

    int w = l.w;
    int h = l.h;
    int batch = l.batch;
    int num_parts = last_l.num_parts;
    int size = last_l.size;

    int total = w*h*batch;

    memset(last_l.seen, 0,l.w*l.h*l.w*l.h*batch * sizeof(float));

    cuda_push_array(last_l.seen_gpu, last_l.seen, l.w*l.h*l.w*l.h*batch);

    update_index_gpu<<<cuda_gridsize(total), BLOCK>>>(last_l.seen_gpu, last_l.delta_sum_gpu, last_l.part_index_gpu, h, w, batch, size, num_parts, total);

}

void update_index(network net, network_state state){

    //fprintf(stderr, "Update index \n");
    layer last_l = net.layers[net.n-1];
    layer l = net.layers[net.n-1 + last_l.pointLayer];

    int i,j,b,p;
    int w = l.w;
    int h = l.h;
    int batch = l.batch;
    int num_parts = last_l.num_parts;
    int size = last_l.size;

    int i_sel = state.i;
    int j_sel = state.j;

    float * part_index = l.part_index_gpu;
    float * delta_sum = l.delta_sum_gpu;

    for(b = 0; b < batch; b++){
        int max = -1;
        int max_i, max_j;
        max_i = -1;
        max_j = -1;

        for(p = 0; p < num_parts; p++){

            for(i = -size + i_sel; i < size + i_sel; i++){
                for(j = -size + j_sel; j < size + j_sel; j++){
                    if((0 <= i < h) && (0 <= j < w)){
                        int out_index = b*w*h + i*w + j;
                        if(delta_sum[out_index] > max){
                            max = delta_sum[out_index];
                            max_i = i;
                            max_j = j;
                        }
                    }
                }
            }
            int part_out_index = (b*w*h*num_parts + p*w*h + i*w + j) * 2;
            part_index[part_out_index + 0] = max_i;
            part_index[part_out_index + 1] = max_j;

            int max_out_index = b*w*h + max_i*w + max_j;
            delta_sum[max_out_index] = -1;

        }
    }

}

void backward_grid_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int lay_sel;
    //fprintf(stderr, "Setting up\n");
    float * original_input = state.input;
    float * original_delta = state.delta;/*

    layer last_l = net.layers[net.n-1];
    int h,w, i, j;

    h = last_l.h; //j
    w = last_l.w; //i

    for(i = 0; i < w; i++){
        for(j = 0; j < h; j++){

            state.i = i;
            state.j = j;*/
            for(lay_sel = net.n-1; lay_sel >= 0; --lay_sel){
                state.index = lay_sel;
                layer l = net.layers[lay_sel];
                //fprintf(stderr, "Layer %d \n",lay_sel);
                if (l.stopbackward) break;
                if(lay_sel == 0){
                    state.input = original_input;
                    state.delta = original_delta;
                }else{
                    layer prev = net.layers[lay_sel-1];
                    state.input = prev.output_gpu;
                    state.delta = prev.delta_gpu;
                }
                l.backward_gpu(l, state);
            }

            update_sum(net);
/*
        }
    }*/
}

void backward_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    //fprintf(stderr, "Setting up\n");
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        //fprintf(stderr, "Layer %d \n",i);
        if (l.stopbackward) break;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        l.backward_gpu(l, state);
    }
}

void update_network_gpu(network net)
{
    cuda_set_device(net.gpu_index);
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    //fprintf(stderr, "Rate = %f, update_batch %d\n",rate, update_batch);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        l.t = get_current_batch(net);
        //fprintf(stderr, "batch = %d\n",l.t);
        if(l.update_gpu){
            //fprintf(stderr, "mentum = %f, decay = %f\n",net.momentum, net.decay);
            l.update_gpu(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

void forward_backward_grid_network_gpu(network net, float *x, float *y)
{

    layer last_l = net.layers[net.n-1];

    int h = last_l.h; //j
    int w = last_l.w; //i

    forward_only_network_gpu(net, x);
    //fprintf(stderr, "Forward\n");

    for(int i = 0; i < w; i++){
        for(int j = 0; j < h; j++){
            network_state state;
            state.i = i;
            state.j = j;
            state.index = 0;
            state.net = net;

            state.delta = 0;
            state.truth = 0;
            state.train = 1;

            backward_grid_network_gpu(net, state);
            //fprintf(stderr, "Backward\n");
        }   
    }
}

void backwardGrad_network_gpu(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    //fprintf(stderr, "Setting up\n");
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        //fprintf(stderr, "Layer %d \n",i);
        if (l.stopbackward) break;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        l.backward_gpu(l, state);
    }
}

void forward_backward_grad_network_gpu(network net, float *x)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
    }
    state.input = *net.input_gpu;
    //state.delta = cuda_make_array(x, x_size);
    state.train = 1;
    state.truth = 0;
    state.delta = 0;
#ifdef CUDNN_HALF
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		cuda_convert_f32_to_f16(l.weights_gpu, l.c*l.n*l.size*l.size, l.weights_gpu16);
	}
#endif
    fprintf(stderr, "Forward\n");
    forward_network_gpu(net, state);
	cudaStreamSynchronize(get_cuda_stream());

    state.delta = cuda_make_array(x, x_size);
    fill_ongpu(x_size, 0, state.delta, 1);
    layer l = net.layers[0];
    fprintf(stderr, "Back\n");
    backwardGrad_network_gpu(net, state);

    cuda_pull_array(state.delta, x, x_size);
    
    cuda_pull_array(l.delta_gpu, l.delta, l.outputs);
    cuda_pull_array(l.output_gpu, l.output, l.outputs);
}

void forward_backward_network_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].truths) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
#ifdef CUDNN_HALF
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		cuda_convert_f32_to_f16(l.weights_gpu, l.c*l.n*l.size*l.size, l.weights_gpu16);
	}
#endif
    forward_network_gpu(net, state);
	cudaStreamSynchronize(get_cuda_stream());
    backward_network_gpu(net, state);
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    *net.seen += net.batch;
    forward_backward_network_gpu(net, x, y);
    float error = get_network_cost(net);
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);

    return error;
}

typedef struct {
    network net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net.gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void pull_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scale_updates) cuda_pull_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_updates(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.n*l.size*l.size*l.c);
        if(l.scale_updates) cuda_push_array(l.scale_updates_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void update_layer(layer l, network net)
{
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    l.t = get_current_batch(net);
    if(l.update_gpu){
        l.update_gpu(l, update_batch, rate, net.momentum, net.decay);
    }
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weights, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scales, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.biases, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.n*l.size*l.size*l.c, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.biases, l.n);
        cuda_pull_array(l.weights_gpu, l.weights, l.n*l.size*l.size*l.c);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.n*l.size*l.size*l.c);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.n*l.size*l.size*l.c);
        if(base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}


void merge_updates(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.n*l.size*l.size*l.c, 1, l.weight_updates, 1, base.weight_updates, 1);
        if (l.scale_updates) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scale_updates, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.bias_updates, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weight_updates, 1);
    }
}

void distribute_updates(layer l, layer base)
{
    if(l.type == CONVOLUTIONAL){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.n);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.n*l.size*l.size*l.c);
        if(base.scale_updates) cuda_push_array(l.scale_updates_gpu, base.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.bias_updates_gpu, base.bias_updates, l.outputs);
        cuda_push_array(l.weight_updates_gpu, base.weight_updates, l.outputs*l.inputs);
    }
}

void sync_layer(network *nets, int n, int j)
{
    //printf("Syncing layer %d\n", j);
    int i;
    network net = nets[0];
    layer base = net.layers[j];
    cuda_set_device(net.gpu_index);
    pull_weights(base);
    for (i = 1; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i].gpu_index);
        layer l = nets[i].layers[j];
        distribute_weights(l, base);
    }
    //printf("Done syncing layer %d\n", j);
}

typedef struct{
    network *nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network *nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network *nets, int n, int interval)
{
    int j;
    int layers = nets[0].n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *nets[0].seen += interval * (n-1) * nets[0].batch * nets[0].subdivisions;
    for (j = 0; j < n; ++j){
        *nets[j].seen = *nets[0].seen;
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network *nets, int n, data d, int interval)
{
    int i;
    int batch = nets[0].batch;
    int subdivisions = nets[0].subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        //printf("%f\n", errors[i]);
        sum += errors[i];
    }
    //cudaDeviceSynchronize();
    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    //cudaDeviceSynchronize();
    free(threads);
    free(errors);
    return (float)sum/(n);
}

float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    if(l.type != REGION) cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
	if (net.gpu_index != cuda_get_device())
		cuda_set_device(net.gpu_index);
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = cuda_make_array(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network_gpu(net, state);
    float *out = get_network_output_gpu(net);
    cuda_free(state.input);
    return out;
}

void forward_backward_joint_network_gpu(network net, network contNet, network joinNet, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = joinNet;
    state.objNet = net;
    state.contNet = contNet;
    int y_size = get_network_output_size(joinNet)*joinNet.batch;
    if(joinNet.layers[joinNet.n-1].truths) y_size = joinNet.layers[joinNet.n-1].truths*joinNet.batch;
    if(!*joinNet.truth_gpu){
        *joinNet.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*joinNet.truth_gpu, y, y_size);
    }
    state.delta = 0;
    state.truth = *joinNet.truth_gpu;
    state.train = 1;
#ifdef CUDNN_HALF
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		cuda_convert_f32_to_f16(l.weights_gpu, l.c*l.n*l.size*l.size, l.weights_gpu16);
	}
#endif
    forward_network_gpu(joinNet, state);
	cudaStreamSynchronize(get_cuda_stream());
    backward_network_gpu(joinNet, state);
}

float train_network_joint_datum_gpu(network net, network contNet, network joinNet, float *x, float *y, float *x_f, float *y_f)
{
    forward_only_network_gpu(net, x);
    forward_only_network_gpu(contNet, x_f);
    *joinNet.seen += joinNet.batch;
    forward_backward_joint_network_gpu(net, contNet, joinNet, x, y);
    //fprintf(stderr, "CHECK Joint\n");
    float error = get_network_cost(joinNet);
    if (((*joinNet.seen) / joinNet.batch) % joinNet.subdivisions == 0) {
        update_network_gpu(joinNet);
    }
    return error;
}