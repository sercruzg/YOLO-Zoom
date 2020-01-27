#ifndef UNMAXPOOL_LAYER_H
#define UNMAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer unmaxpool_layer;

image get_unmaxpool_image(unmaxpool_layer l);
unmaxpool_layer make_unmaxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);
void resize_unmaxpool_layer(unmaxpool_layer *l, int w, int h);
void forward_unmaxpool_layer(const unmaxpool_layer l, network_state state);
void backward_unmaxpool_layer(const unmaxpool_layer l, network_state state);

#ifdef GPU
void forward_unmaxpool_layer_gpu(unmaxpool_layer l, network_state state);
void backward_unmaxpool_layer_gpu(unmaxpool_layer l, network_state state);
#endif

#endif

