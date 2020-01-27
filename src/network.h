// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include <stdint.h>
#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "image.h"
#include "data.h"
#include "tree.h"

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    float *gt;
    float *workspace;
    int gtMax;
    int n;
    int batch;
	int *seen;
    int *gt_seen;
    float epoch;
    int subdivisions;
    float momentum;
    float decay;
    layer *layers;
    int outputs;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;
    int div;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int h, w, c;
    int max_crop;
    int min_crop;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int flip;

    int gpu_index;
    tree *hierarchy;

    #ifdef GPU
    float **input_gpu;
    float **truth_gpu;
	int wait_stream;
    #endif
} network;

typedef struct network_state {
    float *truth;
    float *input;
    float *delta;
    float *workspace;
    int train;
    int index;
    int i,j;
    network net, objNet, contNet, joinNet;
} network_state;

#ifdef GPU
float train_networks(network *nets, int n, data d, int interval);
void sync_nets(network *nets, int n, int interval);
float train_network_datum_gpu(network net, float *x, float *y);
float *network_predict_gpu(network net, float *input);
float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float *get_network_output_gpu(network net);
void forward_network_gpu(network net, network_state state);
void backward_network_gpu(network net, network_state state);
void update_network_gpu(network net);
float train_network_joint_datum_gpu(network net, network contNet, network joinNet, float *x, float *y, float *x_f, float *y_f);
#endif

float get_current_rate(network net);
int get_current_batch(network net);
void free_network(network net);
void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net, network_state state);
void backward_network(network net, network_state state);
void update_network(network net);

float train_network(network net, data d);
float train_network_batch(network net, data d, int n);
float train_network_sgd(network net, data d, int n);
float train_network_datum(network net, float *x, float *y);
float train_joint_network(network net, network contNet, network join, data d, data d_f);
float train_joint_network_datum(network net, network contNet, network joinNet, float *x, float *y, float *x_f, float *y_f);

matrix network_predict_data(network net, data test);
float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
int get_network_input_size(network net);
float get_network_cost(network net);

int get_network_nuisance(network net);
int get_network_background(network net);

#ifdef __cplusplus
}
#endif

#endif

