// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnx/defs/schema.h"
#include "xnnpack_onnx_schema.h"

namespace onnxruntime {
namespace xnnpack {
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackConvolution2d);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackDepthwiseConvolution2d);

class OpSet_XnnPack_ver1 {
 public:
  static void ForEachSchema(std::function<void(ONNX_NAMESPACE::OpSchema&&)> fn) {
    fn(GetOpSchema<class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackConvolution2d)>());    
    fn(GetOpSchema<class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(XnnPack, 1, XnnPackDepthwiseConvolution2d)>());    
  }
};
}  // namespace contrib
}  // namespace onnxruntime