// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xnnpack_onnx_defs.h"
#include <onnx/defs/schema.h>

using namespace onnx;

namespace onnxruntime {
namespace xnnpack {

using ::ONNX_NAMESPACE::Common::StatusCategory;
using ::ONNX_NAMESPACE::Common::StatusCode;

static OnnxStatus ComputeOutputSizeSame(ptrdiff_t input_size, int stride, ptrdiff_t* output_size) {
  if (stride == 0) {
    *output_size = -1;
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  *output_size = input_size + stride - 1;
  *output_size = *output_size / stride;
  return ::ONNX_NAMESPACE::Common::Status::OK();
}

static OnnxStatus ComputeOutputSizeValid(ptrdiff_t input_size, int stride, ptrdiff_t filter_size,
                                         ptrdiff_t* output_size) {
  if (stride == 0) {
    *output_size = -1;
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  if (input_size + 1 <= filter_size) {
    *output_size = -1;
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }
  *output_size = input_size - filter_size + stride;
  *output_size = *output_size / stride;
  return ::ONNX_NAMESPACE::Common::Status::OK();
}

// padding_mode: 0, valid. 1, same
static OnnxStatus ConvShapeInference(ptrdiff_t batch_shape, ptrdiff_t in_height, ptrdiff_t in_width,
                                     ptrdiff_t in_channels, ptrdiff_t out_channels, ptrdiff_t filter_height,
                                     ptrdiff_t filter_width, ptrdiff_t in_channels1, uint32_t strides_h,
                                     uint32_t strides_w, int padding_mode, ptrdiff_t* output0, ptrdiff_t* output1,
                                     ptrdiff_t* output2, ptrdiff_t* output3) {
  if (in_channels != in_channels1) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }

  *output0 = batch_shape;
  if (padding_mode == 1) {
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeSame(in_height, strides_h, output1));
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeSame(in_width, strides_w, output2));
  } else if (padding_mode == 0) {
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeValid(in_height, strides_h, filter_height, output1));
    if (*output1 < 0) {
      return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
    }
    ONNX_RETURN_IF_ERROR(ComputeOutputSizeValid(in_width, strides_w, filter_width, output2));
    if (*output2 < 0) {
      return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
    }
  } else {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL);
  }

  *output3 = out_channels;
  return ::ONNX_NAMESPACE::Common::Status::OK();
}

OnnxStatus XnnPackConvShapeInferImpl(::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                                     ::ONNX_NAMESPACE::TensorShapeProto& weight_shape,
                                     uint32_t input_padding_top, uint32_t input_padding_right,
                                     uint32_t input_padding_bottom, uint32_t input_padding_left,
                                     uint32_t subsampling_height, uint32_t subsampling_width, int padding_mode,
                                     ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape) {
  if (input_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Input tensor must have 4 dimensions");
  }

  if (weight_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Weight tensor must have 4 dimensions");
  }
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
  ONNX_RETURN_IF_ERROR(ConvShapeInference(input_N, input_H, input_W, input_C, out_channels, filter_height, filter_width,
                                          in_channels, subsampling_height, subsampling_width, padding_mode,
                                          &output_shape[0], &output_shape[1], &output_shape[2], &output_shape[3]));
  final_output_shape->add_dim()->set_dim_value(output_shape[0]);
  final_output_shape->add_dim()->set_dim_value(output_shape[1]);
  final_output_shape->add_dim()->set_dim_value(output_shape[2]);
  final_output_shape->add_dim()->set_dim_value(output_shape[3]);
  return OnnxStatus::OK();
}

OnnxStatus XnnPackDepthwiseConvolution2dShapeInferImpl(::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                                                       ::ONNX_NAMESPACE::TensorShapeProto& weight_shape,
                                                       uint32_t input_padding_top, uint32_t input_padding_right,
                                                       uint32_t input_padding_bottom,
                                                       uint32_t input_padding_left, uint32_t subsampling_height,
                                                       uint32_t subsampling_width, int padding_mode,
                                                       ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape) {
  if (input_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Input tensor must have 4 dimensions");
  }

  if (weight_shape.dim_size() != 4) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Weight tensor must have 4 dimensions");
  }


  int64_t input_N = input_shape.dim(0).dim_value();
  int64_t input_H = input_shape.dim(1).dim_value();
  int64_t input_W = input_shape.dim(2).dim_value();
  int64_t input_C = input_shape.dim(3).dim_value();

  if (input_C == 0) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "Input channel can not be zero");
  }

  // Weight shape: [1, kernel_height, kernel_width, input_channels * depth_multiplier]
  int64_t size_one = weight_shape.dim(0).dim_value();
  if (size_one != 1) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL, "The first dim of weight must be one.");
  }
  int64_t filter_height = weight_shape.dim(1).dim_value();
  int64_t filter_width = weight_shape.dim(2).dim_value();
  int64_t input_channels_by_depth_multiplier = weight_shape.dim(3).dim_value();
  if (input_channels_by_depth_multiplier % input_C != 0) {
    return OnnxStatus(StatusCategory::NONE, StatusCode::FAIL,
                      "The last dim of weight is not multiple of input channels.");
  }

  input_H += input_padding_top + input_padding_bottom;
  input_W += input_padding_right + input_padding_left;
  ptrdiff_t output_shape[4];
  ONNX_RETURN_IF_ERROR(ConvShapeInference(input_N, input_H, input_W, input_C, input_channels_by_depth_multiplier,
                                          filter_height, filter_width, input_C, subsampling_height, subsampling_width,
                                          padding_mode, &output_shape[0], &output_shape[1], &output_shape[2],
                                          &output_shape[3]));
  final_output_shape->add_dim()->set_dim_value(output_shape[0]);
  final_output_shape->add_dim()->set_dim_value(output_shape[1]);
  final_output_shape->add_dim()->set_dim_value(output_shape[2]);
  final_output_shape->add_dim()->set_dim_value(output_shape[3]);
  return OnnxStatus::OK();
}

// Compare to the signatures of xnn_define_convolution_2d function, this schema doesn't have
// 1. kernel_height. Because it is just a dimension size of the weights
// 2. kernel_width. Because it is just a dimension size of the weights
// 3. group_input_channels. number of input channels per group. Can be calculated if input channels and the number of
// groups are known
// 4. group_output_channels. As the above
ONNX_XNNPACK_OPERATOR_SET_SCHEMA(
    XnnPackConvolution2d, 1,
    OpSchema()
        .Input(0, "X", "", "tensor(float)")
        .Input(1, "W", "", "tensor(float)")
        .Input(2, "B", "", "tensor(float)")
        .Output(0, "X1", "", "tensor(float)")
        .Attr("input_padding_top", "Implicit zero-padding above 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_right",
              "Implicit zero-padding to the right of 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_bottom", "Implicit zero-padding below 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_left",
              "Implicit zero-padding to the left of 2D input data. Must be 0 if padding mode is SAME",
              AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("subsampling_height", "subsampling_height. TFLite stride_height", AttributeProto::INT)
        .Attr("subsampling_width", "subsampling_width. TFLite stride_width", AttributeProto::INT)
        .Attr("dilation_height", "dilation_height. TFLite dilation_height_factor", AttributeProto::INT)
        .Attr("dilation_width", "dilation_width. TFLite dilation_width_factor", AttributeProto::INT)
        .Attr("groups", "groups", AttributeProto::INT)
        //.Attr("group_input_channels", "group_input_channels", AttributeProto::INT)
        //.Attr("group_output_channels", "group_output_channels", AttributeProto::INT)
        .Attr("padding_mode", "0:VALID. 1:SAME.", AttributeProto::INT)
        .Attr("output_min", "output_min", AttributeProto::FLOAT, false)
        .Attr("output_max", "output_max", AttributeProto::FLOAT, false)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto input_shape = ctx.getInputType(0)->tensor_type().shape();
          auto weight_shape = ctx.getInputType(1)->tensor_type().shape();
          uint32_t input_padding_top = static_cast<uint32_t>(getAttribute(ctx, "input_padding_top", 0));
          uint32_t input_padding_right = static_cast<uint32_t>(getAttribute(ctx, "input_padding_right", 0));
          uint32_t input_padding_bottom = static_cast<uint32_t>(getAttribute(ctx, "input_padding_bottom", 0));
          uint32_t input_padding_left = static_cast<uint32_t>(getAttribute(ctx, "input_padding_left", 0));

          uint32_t subsampling_height = static_cast<uint32_t>(getAttribute(ctx, "subsampling_height", 0));
          uint32_t subsampling_width = static_cast<uint32_t>(getAttribute(ctx, "subsampling_width", 0));
          int padding_mode = static_cast<int>(getAttribute(ctx, "padding_mode", 0));

          auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          OnnxStatus status = XnnPackConvShapeInferImpl(
              input_shape, weight_shape, input_padding_top, input_padding_right, input_padding_bottom,
              input_padding_left, subsampling_height, subsampling_width, padding_mode, final_output_shape);
          if (!status.IsOK()) {
            // Convert the status to an exception
            fail_shape_inference(status.ErrorMessage());
          }
        }));

// Compare to the signatures of xnn_define_convolution_2d function, this schema doesn't have
// 1. kernel_height. Because it is just a dimension size of the weights
// 2. kernel_width. Because it is just a dimension size of the weights
// 3. group_input_channels. number of input channels per group. Can be calculated if input channels and the number of
// groups are known
// 4. group_output_channels. As the above
// 5. depth_multiplier
// Please note this operator uses a different weight layout compared to the normal Convolution2d.
ONNX_XNNPACK_OPERATOR_SET_SCHEMA(
    XnnPackDepthwiseConvolution2d, 1,
    OpSchema()
        .Input(0, "X", "", "tensor(float)")
        .Input(1, "W", "Shape:[1, kernel_height, kernel_width, input_channels * depth_multiplier]", "tensor(float)")
        .Input(2, "B", "", "tensor(float)")
        .Output(0, "X1", "", "tensor(float)")
        .Attr("input_padding_top", "input_padding_top", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_right", "input_padding_right", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_bottom", "input_padding_bottom", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("input_padding_left", "input_padding_left", AttributeProto::INT, static_cast<int64_t>(0))
        //.Attr("kernel_height", "kernel_height", AttributeProto::INT) //TODO: is it just a dim of W?
        //.Attr("kernel_width", "kernel_width", AttributeProto::INT)//TODO: is it just a dim of W?
        .Attr("subsampling_height", "subsampling_height. TFLite stride_height", AttributeProto::INT)
        .Attr("subsampling_width", "subsampling_width. TFLite stride_width", AttributeProto::INT)
        .Attr("dilation_height", "dilation_height. TFLite dilation_height_factor", AttributeProto::INT)
        .Attr("dilation_width", "dilation_width. TFLite dilation_width_factor", AttributeProto::INT)
        .Attr("padding_mode", "0:VALID. 1:SAME.", AttributeProto::INT)
        .Attr("output_min", "output_min", AttributeProto::FLOAT, false)
        .Attr("output_max", "output_max", AttributeProto::FLOAT, false)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto input_shape = ctx.getInputType(0)->tensor_type().shape();
          auto weight_shape = ctx.getInputType(1)->tensor_type().shape();
          uint32_t input_padding_top = static_cast<uint32_t>(getAttribute(ctx, "input_padding_top", 0));
          uint32_t input_padding_right = static_cast<uint32_t>(getAttribute(ctx, "input_padding_right", 0));
          uint32_t input_padding_bottom = static_cast<uint32_t>(getAttribute(ctx, "input_padding_bottom", 0));
          uint32_t input_padding_left = static_cast<uint32_t>(getAttribute(ctx, "input_padding_left", 0));

          uint32_t subsampling_height = static_cast<uint32_t>(getAttribute(ctx, "subsampling_height", 0));
          uint32_t subsampling_width = static_cast<uint32_t>(getAttribute(ctx, "subsampling_width", 0));
          int padding_mode = static_cast<int>(getAttribute(ctx, "padding_mode", 0));

          auto final_output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          OnnxStatus status = XnnPackConvShapeInferImpl(
              input_shape, weight_shape, input_padding_top, input_padding_right, input_padding_bottom,
              input_padding_left, subsampling_height, subsampling_width, padding_mode, final_output_shape);
          if (!status.IsOK()) {
            // Convert the status to an exception
            fail_shape_inference(status.ErrorMessage());
          }
        }));
}  // namespace xnnpack
}  // namespace onnxruntime
