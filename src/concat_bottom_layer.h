#ifndef CONCAT_BOTTOM_LAYER_H
#define CONCAT_BOTTOM_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer concat_bottom_layer;

concat_bottom_layer make_concat_bottom_layer(int batch, int h, int w, int c, int l, int flip, int h_bot, int w_bot, int c_bot, int h_step, int w_step);

void forward_concat_bottom_layer(concat_bottom_layer layer, network_state state);
void backward_concat_bottom_layer(concat_bottom_layer layer, network_state state);

#ifdef GPU
void forward_concat_bottom_layer_gpu(concat_bottom_layer layer, network_state state) ;
void backward_concat_bottom_layer_gpu(concat_bottom_layer layer, network_state state);
#endif

#endif

