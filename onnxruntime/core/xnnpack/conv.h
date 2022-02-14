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
template <typename T>
class Conv : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  ConvAttributes conv_attrs_;
};

template <>
class Conv<float> : public OpKernel {
 public:
  Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    const Tensor* weight = nullptr;
    const Tensor* B = nullptr;
    const ONNX_NAMESPACE::TypeProto* input_type_proto = info.GetInputType(0);
    const ONNX_NAMESPACE::TypeProto* output_type_proto = info.GetOutputType(0);
    output_shape = utils::GetTensorShapeFromTensorShapeProto(output_type_proto->tensor_type().shape());
    ORT_ENFORCE(input_type_proto != nullptr);
    int64_t input_channels = input_type_proto->tensor_type().shape().dim(3).dim_value();
    int64_t output_channels = output_type_proto->tensor_type().shape().dim(3).dim_value();
    ORT_ENFORCE(input_channels != 0 && input_channels % conv_attrs_.group == 0);
    ORT_ENFORCE(output_channels != 0 && output_channels % conv_attrs_.group == 0);
    ORT_ENFORCE(info.TryGetConstantInput(1, &weight));
    ORT_ENFORCE(info.TryGetConstantInput(2, &B));

    
    //bool is_depthwise = conv_attrs_.group == input_channels;
  

    const auto& W_shape = weight->Shape();
    std::vector<int64_t> input_perm = onnx_layout_transformation::ChannelLastToFirstPerm(4);
    TensorShape W_shape_new({4, 4, 4, 4});
    for (size_t i = 0; i != 4; ++i) {
      W_shape_new[i] = W_shape[input_perm[i]];
    }
    ORT_ENFORCE(conv_attrs_.pads.size() == 4);
    ORT_ENFORCE(conv_attrs_.kernel_shape_specified);
    TensorShape attr_weight;
    TensorShapeVector attr_kernel;
    ORT_ENFORCE(conv_attrs_.ComputeKernelShape(W_shape_new, attr_kernel).IsOK());
    ORT_ENFORCE(attr_kernel.size() == 2);
    ORT_ENFORCE(conv_attrs_.strides.size() == 2);
    ORT_ENFORCE(conv_attrs_.dilations.size() == 2);
    //TODO: which is height, which is width?
    ORT_ENFORCE(attr_kernel[0] == attr_kernel[1]);
    ORT_ENFORCE(conv_attrs_.strides[0] == conv_attrs_.strides[1]);
    ORT_ENFORCE(conv_attrs_.dilations[0] == conv_attrs_.dilations[1]);

    //0
    int64_t x1_begin = conv_attrs_.pads[0];
    //0
    int64_t x2_begin = conv_attrs_.pads[1];
    //1
    int64_t x1_end = conv_attrs_.pads[2];
    //1
    int64_t x2_end = conv_attrs_.pads[3];
    
    xnn_status status;
    status = xnn_create_convolution2d_nhwc_f32(
        gsl::narrow<uint32_t>(x1_begin) /* top padding */, gsl::narrow<uint32_t>(x2_end) /* right padding */,
        gsl::narrow<uint32_t>(x1_end) /* bottom padding */, gsl::narrow<uint32_t>(x2_begin) /* left padding */,
        gsl::narrow<uint32_t>(attr_kernel[1]) /* kernel height */, gsl::narrow<uint32_t>(attr_kernel[0]) /* kernel width */,
        gsl::narrow<uint32_t>(conv_attrs_.strides[1]) /* subsampling height */, gsl::narrow<uint32_t>(conv_attrs_.strides[0]) /* subsampling width */,
        gsl::narrow<uint32_t>(conv_attrs_.dilations[1]) /* dilation_height */, gsl::narrow<uint32_t>(conv_attrs_.dilations[0]) /* dilation_width */,
        gsl::narrow<uint32_t>(conv_attrs_.group) /* groups */,
        gsl::narrow<uint32_t>(input_channels / conv_attrs_.group) /* input channels per group */,
        gsl::narrow<uint32_t>(output_channels / conv_attrs_.group) /* output_channels_per_group */,
        gsl::narrow<uint32_t>(input_channels * conv_attrs_.strides[0]) /* input pixel stride */,
        gsl::narrow<uint32_t>(output_channels * conv_attrs_.strides[0]) /* output pixel stride */,
        weight->Data<float>(), B->Data<float>(),
        0.0f /* output min */, 6.0f /* output max */,
        0 /* flags */,
        &op0);
     ORT_ENFORCE(status == xnn_status_success);
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  ConvAttributes conv_attrs_;
  xnn_operator_t op0 = nullptr;
  TensorShape output_shape;
};
}  // namespace xnnpack
}  // namespace onnxruntime
