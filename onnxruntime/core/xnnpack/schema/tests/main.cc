#include "klee/klee.h"
#include <iostream>
#include <sstream>
#include <onnx/common/status.h>
#include "onnx/common/common.h"
#include "onnx/onnx_pb.h"



struct TensorShapeProtoDimension{
  int64_t dim_value;
  char dim_param[2];
};

::ONNX_NAMESPACE::TensorShapeProto CreateTensorShapeProto(TensorShapeProtoDimension* info, size_t len){
  ::ONNX_NAMESPACE::TensorShapeProto ret;
  for(size_t i=0;i!=len;++i){
    TensorShapeProtoDimension& p = info[i];
    if(p.dim_param[0] != 0){
      //std::string s;
      //s.append(1, p.dim_param[0]);
      p.dim_param[1] = '\0';
      ret.add_dim()->set_dim_param(p.dim_param);
      continue;
    }
    if(p.dim_value != 0){
      ret.add_dim()->set_dim_value(p.dim_value);
    }
  }
  return ret;
}

#define MAX_DIM_COUNT 4

int main(){
  TensorShapeProtoDimension data[MAX_DIM_COUNT*2];
  uint8_t dim_len_1, dim_len_2;
  
  klee_make_symbolic(&dim_len_1, sizeof(dim_len_1), "dim_len_1");
  klee_make_symbolic(&dim_len_2, sizeof(dim_len_2), "dim_len_2");
  klee_make_symbolic(&data, sizeof(data), "dim1");

  uint32_t input_padding_top;
  uint32_t input_padding_right;
  uint32_t input_padding_bottom;
  uint32_t input_padding_left;
  
  uint32_t strides_h;
  uint32_t strides_w;
  int padding_mode;
  ptrdiff_t output0;
  ptrdiff_t output1;
  ptrdiff_t output2;
  ptrdiff_t output3;
  klee_make_symbolic(&input_padding_top, sizeof(input_padding_top), "input_padding_top");
  klee_make_symbolic(&input_padding_right, sizeof(input_padding_right), "input_padding_right");
  klee_make_symbolic(&input_padding_bottom, sizeof(input_padding_bottom), "input_padding_bottom");
  klee_make_symbolic(&input_padding_left, sizeof(input_padding_left), "input_padding_left");
  
  klee_make_symbolic(&strides_h, sizeof(strides_h), "strides_h");
  klee_make_symbolic(&strides_w, sizeof(strides_w), "strides_w");
  klee_make_symbolic(&padding_mode, sizeof(padding_mode), "padding_mode");
  ::ONNX_NAMESPACE::TensorShapeProto input_shape,weight_shape, output;
  input_shape = CreateTensorShapeProto(data, 4);
  weight_shape = CreateTensorShapeProto(data + 4, 4);
  
  infer(input_shape, weight_shape, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, strides_h, strides_w, padding_mode, &output);

  return 0;
}

#if 0
int main() {
  int a;
  ptrdiff_t batch_shape;
ptrdiff_t in_height;
ptrdiff_t in_width;
ptrdiff_t in_channels;
ptrdiff_t out_channels;
ptrdiff_t filter_height;
ptrdiff_t filter_width;
ptrdiff_t in_channels1;
uint32_t strides_h;
uint32_t strides_w;
int padding_mode;
ptrdiff_t output0;
ptrdiff_t output1;
ptrdiff_t output2;
 ptrdiff_t output3;
  klee_make_symbolic(&batch_shape, sizeof(batch_shape), "batch_shape");
klee_make_symbolic(&in_height, sizeof(in_height), "in_height");
klee_make_symbolic(&in_width, sizeof(in_width), "in_width");
klee_make_symbolic(&in_channels, sizeof(in_channels), "in_channels");
klee_make_symbolic(&out_channels, sizeof(out_channels), "out_channels");
klee_make_symbolic(&filter_height, sizeof(filter_height), "filter_height");
klee_make_symbolic(&filter_width, sizeof(filter_width), "filter_width");
klee_make_symbolic(&in_channels1, sizeof(in_channels1), "in_channels1");
klee_make_symbolic(&strides_h, sizeof(strides_h), "strides_h");
klee_make_symbolic(&strides_w, sizeof(strides_w), "strides_w");
klee_make_symbolic(&padding_mode, sizeof(padding_mode), "padding_mode");
klee_make_symbolic(&output0, sizeof(output0), "output0");
klee_make_symbolic(&output1, sizeof(output1), "output1");
klee_make_symbolic(&output2, sizeof(output2), "output2");
klee_make_symbolic(&output3, sizeof(output3), "output3");
 ConvShapeInference(batch_shape, in_height, in_width, in_channels, out_channels, filter_height, filter_width, in_channels1, strides_h, strides_w, padding_mode, &output0, &output1, &output2, &output3);
  
  return 0;
}
#endif
