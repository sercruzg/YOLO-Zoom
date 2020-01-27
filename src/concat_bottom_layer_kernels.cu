#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#pragma comment(lib, "cudnn.lib")  
#endif

extern "C" {
#include "concat_bottom_layer.h"
#include "blas.h"
#include "utils.h"
#include "cuda.h"
}

__global__ void apply_concat_bottom_gpu(float* data_im, float* data_imContext, float* data_imJoint,
        const int height, const int width, const int channel, const int size, const int flip, const int h_bot, const int w_bot, const int c_bot,
        const int h_step, const int w_step) {

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if (index >= size)
        return;

    int col = (index % width);
    index = (index / width);
    int row = (index % height);
    index = (index / height);

    int c_out = index;

    if(index < channel){
        data_imJoint[col + width*(row + height*c_out)]  = data_im[col + width*(row + height*c_out)];
    }
    else{
        index = index - channel;
        int h_step_out = index % h_step;
        index = (index / h_step);
        int w_step_out = index % w_step;
        int cur_c = index / w_step;
        int cur_w = col*w_step + w_step_out;
        int cur_h = row*h_step + h_step_out;

        int min_h = (height*h_step) - h_bot;
        if(cur_h < min_h){
            data_imJoint[col + width*(row + height*c_out)]  = 0;
        }
        else{
            int cur_h_bot = cur_h - min_h;
            //printf("cur_h= %d,  min_h=%d cur_w=%d, cur_c=%d\n",cur_h, min_h, cur_w, cur_c);
            //float value1 = data_imContext[cur_h_bot + width*((w_bot-1) + height*cur_c)];
            //float value2 = data_imContext[(h_bot-1) + width*(cur_w + height*cur_c)];
            //float value3 = data_imContext[(h_bot-1) + width*((w_bot-1) + height*cur_c)];

            float value = data_imContext[cur_w + w_bot*(cur_h_bot + h_bot*cur_c)];
            data_imJoint[col + width*(row + height*c_out)] = value;
        }
    }

}

void forward_concat_bottom_layer_gpu(concat_bottom_layer l, network_state state) {

    float *objInput = state.objNet.layers[state.objNet.n - 1 + l.pointLayer].output_gpu;

    int obj_h = state.objNet.layers[state.objNet.n - 1 + l.pointLayer].out_h;
    int obj_w = state.objNet.layers[state.objNet.n - 1 + l.pointLayer].out_w;
    int obj_c = state.objNet.layers[state.objNet.n - 1 + l.pointLayer].out_c;

    int cont_h = state.contNet.layers[state.contNet.n - 1 + l.pointLayer].out_h;
    int cont_w = state.contNet.layers[state.contNet.n - 1 + l.pointLayer].out_w;
    int cont_c = state.contNet.layers[state.contNet.n - 1 + l.pointLayer].out_c;
    float *contInput = state.contNet.layers[state.contNet.n - 1 + l.pointLayer].output_gpu;

    //fprintf(stderr, "N= %d,  point=%d select=%d\n",state.objNet.n, l.pointLayer,state.objNet.n - 1 + l.pointLayer);
    int total = l.out_h * l.out_w * l.out_c;
    int flip = l.flip;
    for(int i = 0; i < l.batch; i++){
        float *a = objInput + i*obj_c*obj_h*obj_w;
        float *b = contInput + i*cont_c*cont_h*cont_w;
        float *c = l.output_gpu + i*l.out_h*l.out_w*l.out_c;
        apply_concat_bottom_gpu<<<cuda_gridsize(total), BLOCK>>>(a, b, c, obj_h, obj_w, obj_c, total, flip, cont_h, cont_w, cont_c, l.h_step, l.w_step);
    }

}
void backward_concat_bottom_layer_gpu(concat_bottom_layer l, network_state state)
{
}