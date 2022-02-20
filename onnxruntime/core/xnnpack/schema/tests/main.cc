#include "klee/klee.h"
#include <iostream>
#include <sstream>
#include <onnx/common/status.h>
#include "onnx/common/common.h"
#include "onnx/onnx_pb.h"
#include "xnnpack_onnx_defs.h"

using namespace onnxruntime::xnnpack;

namespace onnxruntime {
namespace xnnpack {
OnnxStatus ConvShapeInference(const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& batch_shape, ptrdiff_t in_height, ptrdiff_t in_width,
                              ptrdiff_t in_channels, const ::ONNX_NAMESPACE::TensorShapeProto_Dimension& out_channels, ptrdiff_t filter_height,
                              ptrdiff_t filter_width, ptrdiff_t in_channels1, uint32_t strides_h,
                              uint32_t strides_w, int padding_mode, ::ONNX_NAMESPACE::TensorShapeProto_Dimension** output);
}

}  // namespace onnxruntime

struct TensorShapeProtoDimension{
  int64_t dim_value;
  char dim_param;
  uint8_t one_case;
};

::ONNX_NAMESPACE::TensorShapeProto CreateTensorShapeProto(TensorShapeProtoDimension* info, size_t len){
  ::ONNX_NAMESPACE::TensorShapeProto ret;
  for(size_t i=0;i!=len;++i){
    TensorShapeProtoDimension& p = info[i];
    auto& d = *ret.add_dim();
    if (p.one_case == 0) {
      std::string s(1, p.dim_param);
      d.set_dim_param(s);
      continue;
    }
    if (p.one_case == 1) {
      d.set_dim_value(p.dim_value);
      continue;
    }
  }
  return ret;
}

const ::ONNX_NAMESPACE::TensorShapeProto_Dimension CreateTensorShapeProtoDimension(TensorShapeProtoDimension& p) {
  ::ONNX_NAMESPACE::TensorShapeProto_Dimension ret;
  if (p.one_case == 0) {
    std::string s(1, p.dim_param);
    ret.set_dim_param(s);
    return ret;
  }
  if (p.one_case == 1) {
    ret.set_dim_value(p.dim_value);
    return ret;
  }
  return ret;
}

extern "C" {
int TestConvShapeInference();
}

int TestConvShapeInference() {
  TensorShapeProtoDimension data[2];
  ptrdiff_t in_height;
  ptrdiff_t in_width;
  ptrdiff_t in_channels;
  ptrdiff_t filter_height;
  ptrdiff_t filter_width;
  ptrdiff_t in_channels1;
  ptrdiff_t strides_h;
  ptrdiff_t strides_w;
  int padding_mode;

  klee_make_symbolic(&data, sizeof(data), "dim1");
  klee_make_symbolic(&in_height, sizeof(in_height), "in_height");
  klee_make_symbolic(&in_width, sizeof(in_width), "in_width");
  klee_make_symbolic(&in_channels, sizeof(in_channels), "in_channels");
  klee_make_symbolic(&filter_height, sizeof(filter_height), "filter_height");
  klee_make_symbolic(&filter_width, sizeof(filter_width), "filter_width");
  klee_make_symbolic(&in_channels1, sizeof(in_channels1), "in_channels1");
  klee_make_symbolic(&strides_h, sizeof(strides_h), "strides_h");
  klee_make_symbolic(&strides_w, sizeof(strides_w), "strides_w");
  klee_make_symbolic(&padding_mode, sizeof(padding_mode), "padding_mode");

  ::ONNX_NAMESPACE::TensorShapeProto_Dimension d1 = CreateTensorShapeProtoDimension(data[0]);
  ::ONNX_NAMESPACE::TensorShapeProto_Dimension d2 = CreateTensorShapeProtoDimension(data[1]);
  ::ONNX_NAMESPACE::TensorShapeProto_Dimension output[4];
  ::ONNX_NAMESPACE::TensorShapeProto_Dimension* output_ptrs[4];
  output_ptrs[0] = output;
  output_ptrs[1] = output + 1;
  output_ptrs[2] = output + 2;
  output_ptrs[3] = output + 3;
  ConvShapeInference(d1, in_height, in_width,
                     in_channels, d2, filter_height,
                     filter_width, in_channels1, strides_h,
                     strides_w, padding_mode, output_ptrs);
  return 0;
}
