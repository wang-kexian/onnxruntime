#include <string>
#include <stack>
#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/onnx.h>
#include <ATen/core/functional.h>
#include "core/framework/session_options.h"
#include "orttraining/core/session/training_session.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/ortdevice.h"
#include "core/framework/ort_value.h"
#include "python/onnxruntime_pybind_state_common.h"
// Scalar type translation from ONNX to Pytorch.
c10::ScalarType CreateC10ScalarType(const onnxruntime::PrimitiveDataTypeBase* elem_type);
// Scalar type translation from Pytorch to ORT.
onnxruntime::MLDataType CreateOrtScalarType(at::ScalarType dtype);
// Device translation from Pytorch to ORT.
OrtDevice CreateOrtDevice(const c10::Device device);
// Device translation from ORT to Pytorch.
c10::Device CreateC10Device(const OrtDevice& device);
// Create a tensor from a Pytorch tensor. No memory copy. 
// Conceptually, the returned tensor is a view of the input tensor.
OrtValue CreateOrtTensorValue(const at::Tensor& tensor);
// Similarly, create a Pytorch tensor from an OrtValue without
// memory copy.
// The created at::Tensor and onnxruntime::Tensor have
// the same lifetime.
c10::IValue CreateC10IvalueTensor(OrtValue value);