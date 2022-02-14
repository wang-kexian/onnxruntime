/**
* Copyright (c) 2016-present, Facebook, Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/* Modifications Copyright (c) Microsoft. */

#include "core/xnnpack/conv.h"

#include "core/common/safeint.h"
#include "core/xnnpack/build_kernel_info.h"

#define XNNPACK_CPU_MS_DOMAIN_OPERATOR_KERNEL(name, ver, builder, ...) \
  ONNX_OPERATOR_KERNEL_EX(name, kMSDomain, ver, kCpuExecutionProvider, builder, __VA_ARGS__)

namespace onnxruntime {
namespace xnnpack {


Status Conv<float>::Compute(OpKernelContext* context) const {
  std::cout << "running " << context->GetNodeName() << std::endl;
  const auto* X = context->Input<Tensor>(0); 
  Tensor* Y = context->Output(0, output_shape);
  const TensorShape& input_shape = X->Shape();
  xnn_status status = xnn_setup_convolution2d_nhwc_f32(
      op0,
      input_shape[0] /* batch size */, input_shape[1] /* input height */, input_shape[2] /* input width */,
      X->Data<float>() /* input */, Y->MutableData<float>() /* output */,
      nullptr /* threadpool */);
  ORT_ENFORCE(status == xnn_status_success);
  status = xnn_run_operator(op0, nullptr);
  ORT_ENFORCE(status == xnn_status_success);
  return Status::OK();
}

XNNPACK_CPU_MS_DOMAIN_OPERATOR_KERNEL(
    NhwcConv,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);
}  // namespace xnnpack
}  // namespace onnxruntime
