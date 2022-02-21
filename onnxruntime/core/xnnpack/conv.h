// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/transpose_optimizer/api.h"
#include <xnnpack.h>

namespace onnxruntime {
namespace xnnpack {

class Convolution2d : public OpKernel {
 public:
  Convolution2d(const OpKernelInfo& info) : OpKernel(info) {
    const Tensor* weight = nullptr;
    const Tensor* B = nullptr;
    const ONNX_NAMESPACE::TypeProto* input_type_proto = info.GetInputType(0);
    const ONNX_NAMESPACE::TypeProto* output_type_proto = info.GetOutputType(0);
    ORT_ENFORCE(input_type_proto != nullptr);
    ORT_ENFORCE(output_type_proto != nullptr);
    ORT_ENFORCE(info.TryGetConstantInput(1, &weight));
    ORT_ENFORCE(info.TryGetConstantInput(2, &B));

    int64_t input_channels = input_type_proto->tensor_type().shape().dim(3).dim_value();
    int64_t output_channels = output_type_proto->tensor_type().shape().dim(3).dim_value();
    const TensorShape& kernel_shape = weight->Shape();
    int64_t kernel_height = kernel_shape[1];
    int64_t kernel_width = kernel_shape[2];

    int64_t input_padding_top;
    int64_t input_padding_right;
    int64_t input_padding_bottom;
    int64_t input_padding_left;

    int64_t subsampling_height;
    int64_t subsampling_width;
    int64_t dilation_height;
    int64_t dilation_width;
    int64_t groups;
    float output_min;
    float output_max;
    int64_t padding_mode;
    ORT_ENFORCE(info.GetAttr("input_padding_top", &input_padding_top).IsOK());
    ORT_ENFORCE(info.GetAttr("input_padding_right", &input_padding_right).IsOK());
    ORT_ENFORCE(info.GetAttr("input_padding_bottom", &input_padding_bottom).IsOK());
    ORT_ENFORCE(info.GetAttr("input_padding_left", &input_padding_left).IsOK());
    ORT_ENFORCE(info.GetAttr("subsampling_height", &subsampling_height).IsOK());
    ORT_ENFORCE(info.GetAttr("subsampling_width", &subsampling_width).IsOK());
    ORT_ENFORCE(info.GetAttr("dilation_height", &dilation_height).IsOK());
    ORT_ENFORCE(info.GetAttr("dilation_width", &dilation_width).IsOK());
    ORT_ENFORCE(info.GetAttr("groups", &groups).IsOK());
    // TODO: handle optional case
    ORT_ENFORCE(info.GetAttr("output_min", &output_min).IsOK());
    ORT_ENFORCE(info.GetAttr("output_max", &output_max).IsOK());
    ORT_ENFORCE(info.GetAttr("padding_mode", &padding_mode).IsOK());
    uint32_t flags = 0;
    if (padding_mode == 1) {
      flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
    }
    size_t group_input_channels = input_channels / groups;
    size_t group_output_channels = output_channels / groups;
    xnn_status status;
    status = xnn_create_convolution2d_nhwc_f32(
        gsl::narrow<uint32_t>(input_padding_top),
        gsl::narrow<uint32_t>(input_padding_right),
        gsl::narrow<uint32_t>(input_padding_bottom),
        gsl::narrow<uint32_t>(input_padding_left),
        gsl::narrow<uint32_t>(kernel_height),
        gsl::narrow<uint32_t>(kernel_width),
        gsl::narrow<uint32_t>(subsampling_height),
        gsl::narrow<uint32_t>(subsampling_width),
        gsl::narrow<uint32_t>(dilation_height),
        gsl::narrow<uint32_t>(dilation_width),
        gsl::narrow<uint32_t>(groups),
        gsl::narrow<uint32_t>(group_input_channels),
        gsl::narrow<uint32_t>(group_output_channels),
        gsl::narrow<uint32_t>(input_channels),
        gsl::narrow<uint32_t>(output_channels),
        weight->Data<float>(),
        B->Data<float>(),
        output_min,
        output_max,
        flags,
        &op0);
    ORT_ENFORCE(status == xnn_status_success);
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  xnn_operator_t op0 = nullptr;
  TensorShape output_shape;
};

class DepthWiseConvolution2d : public OpKernel {
 public:
  DepthWiseConvolution2d(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext*) const override {
    return Status::OK();
  }
};
}  // namespace xnnpack
}  // namespace onnxruntime
