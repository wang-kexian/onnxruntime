#include "klee/klee.h"
#include <iostream>
#include <sstream>
#include <onnx/common/status.h>
#include "onnx/common/common.h"
#include "onnx/onnx_pb.h"
#include "xnnpack_onnx_defs.h"

using namespace onnxruntime::xnnpack;

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
  
  XnnPackDepthwiseConvolution2dShapeInferImpl(input_shape, weight_shape, input_padding_top, input_padding_right, input_padding_bottom, input_padding_left, strides_h, strides_w, padding_mode, &output);

  return 0;
}

