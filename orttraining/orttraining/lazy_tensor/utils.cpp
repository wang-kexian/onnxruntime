// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "utils.h"
#include <string>
#include <torch/torch.h>

namespace onnxruntime {
namespace lazytensor {
c10::ScalarType CreateC10ScalarType(const onnxruntime::PrimitiveDataTypeBase* elem_type) {
  ORT_ENFORCE(elem_type, "Element type pointer cannot be NULL.");
  switch (static_cast<ONNX_NAMESPACE::TensorProto_DataType>(elem_type->GetDataType())) {
    case onnxruntime::data_types_internal::ToTensorDataType<float>(): {
      return c10::kFloat;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<double>(): {
      return c10::kDouble;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<onnxruntime::MLFloat16>(): {
      return at::kHalf;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<onnxruntime::BFloat16>(): {
      return c10::kBFloat16;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<bool>(): {
      return at::kBool;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<int16_t>(): {
      return at::kShort;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<int>(): {
      return at::kInt;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<int64_t>(): {
      return at::kLong;
    }
    default:
      ORT_THROW("Unsupport ORT scalar type.");
  }
}

onnxruntime::MLDataType CreateOrtScalarType(
    at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return onnxruntime::DataTypeImpl::GetType<float>();
    case at::kDouble:
      return onnxruntime::DataTypeImpl::GetType<double>();
    case at::kHalf:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::MLFloat16>();
    case at::kBFloat16:
      return onnxruntime::DataTypeImpl::GetType<onnxruntime::BFloat16>();
    case at::kInt:
      return onnxruntime::DataTypeImpl::GetType<int>();
    case at::kShort:
      return onnxruntime::DataTypeImpl::GetType<int16_t>();
    case at::kLong:
      return onnxruntime::DataTypeImpl::GetType<int64_t>();
    case at::kBool:
      return onnxruntime::DataTypeImpl::GetType<bool>();
    default:
      ORT_THROW("Unsupport aten scalar type: ", dtype);
  }
}

OrtDevice CreateOrtDevice(const c10::Device device) {
  // c10::Device's ID can be negative, which means current device.
  // Assumptions:
  //  1. c10::Device::CPU is always indexed by -1.
  //     Thus, it's mapped to OrtDevice::CPU with index 0.
  //  2. c10::Device::GPU always has non-negative index.
  //     If the index is i, it's mapped to OrtDevice::GPU with index i.

  // For each case, assert if our assumptions are true and then do the work.
  if (device.type() == c10::DeviceType::CPU) {
    ORT_ENFORCE(device.index() == -1);
    return OrtDevice(OrtDevice::CPU, OrtDevice::MemType::DEFAULT, 0);
  } else if (device.type() == c10::DeviceType::CUDA) {
    ORT_ENFORCE(device.index() >= 0);
    return OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, device.index());
  } else {
    ORT_THROW("Unsupport Pytorch c10 device.",
              " Type: ", c10::DeviceTypeName(device.type()), ",",
              " ID: ", device.index());
  }
}

c10::Device CreateC10Device(const OrtDevice& device) {
  // Handles CPU, GPU, and throws otherwise.
  switch (device.Type()) {
    case OrtDevice::CPU: {
      ORT_ENFORCE(device.Id() == 0, "ORT CPU device ID must be 0 but got ", device.Id());
      // No need to specify index when creating c10 CPU.
      return c10::Device(c10::DeviceType::CPU);
    }
    case OrtDevice::GPU: {
      ORT_ENFORCE(device.Id() >= 0, "ORT GPU device ID must be >= 0 but got ", device.Id());
      // c10 GPU can have negative index (means current device),
      // but only using non-negative index is enough to cover all ORT cases.
      return c10::Device(c10::DeviceType::CUDA, device.Id());
    }
    default: {
      // Got unsupported device. Throws.
      std::string device_str;
      if (device.Type() == OrtDevice::CPU) {
        device_str = "CPU";
      } else if (device.Type() == OrtDevice::GPU) {
        device_str = "GPU";
      } else {
        device_str = "Unknown";
      }
      ORT_THROW("Unsupport ORT device: ", device_str, ", ID: ", device.Id());
    }
  }
}

OrtValue CreateOrtTensorValue(const at::Tensor& tensor) {
  onnxruntime::MLDataType element_type = CreateOrtScalarType(tensor.scalar_type());
  onnxruntime::TensorShape shape(tensor.sizes().vec());
  OrtDevice device = CreateOrtDevice(tensor.device());
  OrtMemoryInfo memory_info = OrtMemoryInfo("LTC", OrtAllocatorType::OrtDeviceAllocator, device, device.Id());
  // This tensor's life time is controlled by Pytorch.
  // TODO: consider to let ORT also own that tensor.
  std::unique_ptr<onnxruntime::Tensor> ort_tensor = std::make_unique<onnxruntime::Tensor>(
      element_type, shape,
      tensor.data_ptr(), memory_info);

  OrtValue ort_value;
  ort_value.Init(
      ort_tensor.release(),
      onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
      onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
  return ort_value;
}

c10::IValue CreateC10IvalueTensor(OrtValue value) {
  onnxruntime::Tensor* tensor = value.GetMutable<onnxruntime::Tensor>();
  const OrtDevice& device = tensor->Location().device;
  auto options = torch::TensorOptions()
                     .dtype(CreateC10ScalarType(tensor->DataType()->AsPrimitiveDataType()))
                     .layout(torch::kStrided)
                     .device(CreateC10Device(device))
                     .requires_grad(false);

  // Extract shape from onnxruntime::TensorShape as a vector.
  auto create_shape_vector = [](const onnxruntime::TensorShape& shape) {
    std::vector<int64_t> new_shape(shape.NumDimensions());
    shape.CopyDims(new_shape.data(), shape.NumDimensions());
    return new_shape;
  };

  at::Tensor new_tensor = torch::from_blob(
      tensor->MutableDataRaw(),
      create_shape_vector(tensor->Shape()),
      // Capture-by-value means
      //  1. A new OrtValue is direct-initialized from "value".
      //  2. The new OrtValue and "value" share the same underlying tensor, so
      //     the tensor's lifetime is controlled by both of them, whichever is longer.
      //  3. The new OrtValue's lifetime is the same as this lambda function.
      //  4. This lambda function is deleted by "new_tensor"'s dtor, which also ends
      //     the underlying tensor's life.
      [value](void*) {},
      options);

  return c10::IValue(new_tensor);
}
}  // namespace lazytensor
}  // namespace onnxruntime
