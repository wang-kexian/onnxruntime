#include "klee/klee.h"
#include <iostream>
#include <sstream>
#include <onnx/common/status.h>
#include "onnx/common/common.h"
#include "onnx/onnx_pb.h"

#define ONNX_RETURN_IF_ERROR(expr)                                                           \
  do {                                                                                       \
    auto _status = (expr);                                                                   \
    if ((!_status.IsOK())) {                                                                 \
      return _status;                                                                        \
    }                                                                                        \
  } while (0)

using ::ONNX_NAMESPACE::Common::StatusCategory;
using ::ONNX_NAMESPACE::Common::StatusCode;

static ::ONNX_NAMESPACE::Common::Status ComputeOutputSizeSame(ptrdiff_t input_size, int stride, ptrdiff_t* output_size) {
  if(stride==0) {
    *output_size = -1;
    return ::ONNX_NAMESPACE::Common::Status(StatusCategory::NONE, StatusCode::FAIL);
  }
  *output_size = input_size + stride - 1;
  *output_size = *output_size / stride;
  return ::ONNX_NAMESPACE::Common::Status::OK();
}

static ::ONNX_NAMESPACE::Common::Status ComputeOutputSizeValid(ptrdiff_t input_size, int stride, ptrdiff_t filter_size, ptrdiff_t* output_size) {
  if(stride==0) {
    *output_size = -1;
    return ::ONNX_NAMESPACE::Common::Status(StatusCategory::NONE, StatusCode::FAIL);
  }
  if (input_size + 1 <= filter_size) {
    *output_size = -1;
    return ::ONNX_NAMESPACE::Common::Status(StatusCategory::NONE, StatusCode::FAIL);
  }
  *output_size = input_size - filter_size + stride;
  *output_size = *output_size / stride;
  return ::ONNX_NAMESPACE::Common::Status::OK();
}
//padding_mode: 0, valid. 1, same
static ::ONNX_NAMESPACE::Common::Status ConvShapeInference(ptrdiff_t batch_shape, ptrdiff_t in_height, ptrdiff_t in_width, ptrdiff_t in_channels, ptrdiff_t out_channels, ptrdiff_t filter_height, ptrdiff_t filter_width, ptrdiff_t in_channels1, uint32_t strides_h, uint32_t strides_w, int padding_mode, ptrdiff_t* output0, ptrdiff_t* output1, ptrdiff_t* output2, ptrdiff_t* output3) {
  if (in_channels != in_channels1) {
    return ::ONNX_NAMESPACE::Common::Status(StatusCategory::NONE, StatusCode::FAIL);    
  }

  *output0 = batch_shape;
  if (padding_mode == 1) {
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeSame(in_height, strides_h, output1));
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeSame(in_width, strides_w, output2));
  } else if (padding_mode == 0) {
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeValid(in_height, strides_h, filter_height, output1));
    if (*output1 < 0) {
      return ::ONNX_NAMESPACE::Common::Status(StatusCategory::NONE, StatusCode::FAIL);          
    }
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeValid(in_width, strides_w, filter_width, output2));
    if (*output2 < 0) {
      return ::ONNX_NAMESPACE::Common::Status(StatusCategory::NONE, StatusCode::FAIL);          
    }
  } else {
    return ::ONNX_NAMESPACE::Common::Status(StatusCategory::NONE, StatusCode::FAIL);              
  }

  *output3 = out_channels;
  return ::ONNX_NAMESPACE::Common::Status::OK();
}



static void infer(::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                  ::ONNX_NAMESPACE::TensorShapeProto& weight_shape,
                  uint32_t input_padding_top,
                  uint32_t input_padding_right,
                  uint32_t input_padding_bottom,
                  uint32_t input_padding_left,
                  uint32_t subsampling_height,
                  uint32_t subsampling_width,
                  int padding_mode,
		  ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape) {
  if(input_shape.dim_size()!=4) return;
  if(weight_shape.dim_size()!=4) return;
  int64_t input_N = input_shape.dim(0).dim_value();
  int64_t input_H = input_shape.dim(1).dim_value();
  int64_t input_W = input_shape.dim(2).dim_value();
  int64_t input_C = input_shape.dim(3).dim_value();

  int64_t out_channels = weight_shape.dim(0).dim_value();
  int64_t filter_height = weight_shape.dim(1).dim_value();
  int64_t filter_width = weight_shape.dim(2).dim_value();
  int64_t in_channels = weight_shape.dim(3).dim_value();
  input_H += input_padding_top + input_padding_bottom;
  input_W += input_padding_right + input_padding_left;
  ptrdiff_t output_shape[4];
  ConvShapeInference(input_N, input_H, input_W, input_C, out_channels, filter_height, filter_width, in_channels, subsampling_height, subsampling_width, padding_mode,
                     &output_shape[0], &output_shape[1], &output_shape[2], &output_shape[3]);
  final_output_shape->add_dim()->set_dim_value(output_shape[0]);
  final_output_shape->add_dim()->set_dim_value(output_shape[1]);
  final_output_shape->add_dim()->set_dim_value(output_shape[2]);
  final_output_shape->add_dim()->set_dim_value(output_shape[3]);
}


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
