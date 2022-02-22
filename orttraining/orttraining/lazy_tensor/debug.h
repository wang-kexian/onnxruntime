// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <torch/torch.h>

namespace onnxruntime {
namespace lazytensor {
// This function contains function for comparing values
// and printing values. They are mainly used for debugging.

bool CompareTensor(const at::Tensor& left, const at::Tensor& right);
bool CompareScalar(const at::Scalar& left, const at::Scalar& right);
bool Compare(const c10::IValue& left, const c10::IValue& right);
bool CompareStack(const torch::jit::Stack& left, const torch::jit::Stack& right);
std::string ToString(const c10::IValue& value);
std::string ToString(const at::ArrayRef<c10::IValue>& values);
}  // namespace lazytensor
}  // namespace onnxruntime