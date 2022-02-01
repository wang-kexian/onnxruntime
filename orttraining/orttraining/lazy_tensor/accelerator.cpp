#include "accelerator.h"
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

namespace py = pybind11;
namespace aten = torch::jit::aten;
namespace prim = torch::jit::prim;

// static variable used to create inference session and training session.
static std::unique_ptr<onnxruntime::Environment> ltc_env;

void InitializeLtcEnv() {
  auto initialize = [&]() {
    std::cout << "InitializeLtcEnv" << std::endl;
    // Initialization of the module
    static std::string name = std::string("LTC");
    ORT_THROW_IF_ERROR(onnxruntime::Environment::Create(std::make_unique<onnxruntime::logging::LoggingManager>(
                                                  std::make_unique<onnxruntime::logging::CLogSink>(),
                                                  onnxruntime::logging::Severity::kWARNING, false, onnxruntime::logging::LoggingManager::InstanceType::Temporal,
                                                  &name),
                                              ltc_env));
    static bool initialized = false;
    if (initialized) {
      return;
    }
    initialized = true;
  };
  initialize();
}

onnxruntime::Environment& GetLtcEnv() {
  if (!ltc_env) {
    InitializeLtcEnv();
  }
  return *ltc_env;
}

onnxruntime::MLDataType to_ort_scalar_type(
  at::ScalarType dtype) {
  switch (dtype){
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

//OrtDevice::DeviceId create_ort_device_id(const c10::DeviceIndex device_id) {
//  if (device_id < 0) {
//    return 0;
//  } else {
//    return static_cast<OrtDevice::DeviceId>(device_id);
//  }
//};


//OrtDevice create_ort_device(const c10::DeviceType device_type, const c10::DeviceIndex device_id) {
//    // Assume ID is the same in ORT and Pytorch.
//    OrtDevice::DeviceId ort_device_id = create_ort_device_id(device_id);
//    // Translate Pytorch device type to ORT device type.
//    OrtDevice::DeviceType ort_device_type;
//    switch (device_type) {
//        case c10::DeviceType::CPU:
//            ort_device_type = OrtDevice::CPU;
//            break;
//        case c10::DeviceType::CUDA:
//            ort_device_type = OrtDevice::GPU;
//            break;
//        default:
//        ORT_THROW(
//          "Unsupport Pytorch device.",
//          " Type: ", c10::DeviceTypeName(device_type), ",",
//          " ID: ", device_id);
//    };
//
//    // TODO: check if we should always do OrtAllocatorType::OrtDeviceAllocator.
//    return OrtDevice(ort_device_type, OrtDevice::MemType::DEFAULT, ort_device_id);
//}

OrtMemoryInfo create_ort_memory_info(const OrtDevice& device) {
  // TODO:: check if we should always use OrtAllocatorType::OrtDeviceAllocator.
  return OrtMemoryInfo("LazyTensor-bridge", OrtAllocatorType::OrtDeviceAllocator, device, device.Id());
}

//OrtMemoryInfo create_ort_cpu_memory_info(const OrtDevice& device) {
//  // TODO:: check if we should always use OrtAllocatorType::OrtDeviceAllocator.
//  return OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator, device, device.Id());
//}

// Create OrtDevice from c10::Device.
OrtDevice create_ort_device(const c10::Device device) {
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

OrtValue create_ort_tensor_value(const at::Tensor& tensor) {
  onnxruntime::MLDataType element_type = to_ort_scalar_type(tensor.scalar_type());
  onnxruntime::TensorShape shape(tensor.sizes().vec());
  OrtDevice device = create_ort_device(tensor.device());
  OrtMemoryInfo memory_info = create_ort_memory_info(device);
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

// OrtValue create_ort_scalar_value(const at::Scalar& scalar) {
  // onnxruntime::MLDataType element_type = to_ort_scalar_type(scalar.type());
  // onnxruntime::TensorShape shape({});
  // // This tensor's life time is controlled by Pytorch.
  // // TODO: consider to let ORT also own that tensor.
  // void* data_ptr = nullptr;
  // std::function<void()> data_deleter;

  // switch (scalar.type()) {
    // case at::kFloat: {
      // data_ptr = new float;
      // *reinterpret_cast<float*>(data_ptr) = scalar.toFloat();
      // data_deleter = [=]() {
        // delete reinterpret_cast<float*>(data_ptr);
      // };
      // break;
    // }
    // case at::kDouble: {
      // data_ptr = new double;
      // *reinterpret_cast<double*>(data_ptr) = scalar.toDouble();
      // data_deleter = [=]() {
        // delete reinterpret_cast<double*>(data_ptr);
      // };
      // break;
    // }
    // case at::kBFloat16: {
      // at::BFloat16 valBFloat16 = scalar.toBFloat16();
      // Ort::BFloat16_t *valOrtBFloat16 = reinterpret_cast<Ort::BFloat16_t *>(&valBFloat16);
      // data_ptr = new Ort::BFloat16_t;
      // *reinterpret_cast<Ort::BFloat16_t*>(data_ptr) = *valOrtBFloat16;
      // data_deleter = [=]() {
        // delete reinterpret_cast<Ort::BFloat16_t*>(data_ptr);
      // };
      // break;
    // }
    // default:
      // ORT_THROW("Unsupport aten scalar type: ", scalar.type());
  // }

  // OrtMemoryInfo memory_info = create_ort_cpu_memory_info("at::Scalar on CPU");
  // std::unique_ptr<onnxruntime::Tensor> ort_tensor = std::make_unique<onnxruntime::Tensor>(
      // element_type, shape,
      // data_ptr, memory_info);

  // std::function<void(void*)> deleter = [=](void* p) {
   // data_deleter();
   // onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc()(p);
  // };

  // OrtValue ort_value;
  // ort_value.Init(
      // ort_tensor.release(),
      // onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
      // deleter);
  // return ort_value;
// }


c10::ScalarType create_torch_element_type(const onnxruntime::PrimitiveDataTypeBase* elem_type) {
  ORT_ENFORCE(elem_type, "Element type pointer cannot be NULL.");
  std::cout << "c10::ScalarType create_torch_element_type: " << static_cast<ONNX_NAMESPACE::TensorProto_DataType>(elem_type->GetDataType()) << std::endl;
  switch (static_cast<ONNX_NAMESPACE::TensorProto_DataType>(elem_type->GetDataType())) {
    case onnxruntime::data_types_internal::ToTensorDataType<float>() : {
      return c10::kFloat;  
    }
    case onnxruntime::data_types_internal::ToTensorDataType<double>() : {
      return c10::kDouble;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<onnxruntime::MLFloat16>() : {
      return at::kHalf;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<onnxruntime::BFloat16>() : {
      return c10::kBFloat16;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<bool>() : {
      return at::kBool;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<int16_t>() : {
      return at::kShort;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<int>() : {
      return at::kInt;
    }
    case onnxruntime::data_types_internal::ToTensorDataType<int64_t>() : {
      return at::kLong;
    }
    default:
      ORT_THROW("Unsupport ORT scalar type.");
  }
}

c10::Device create_c10_device(const OrtDevice& device) {
  if (device.Type() == OrtDevice::CPU) {
    ORT_ENFORCE(device.Id() == 0, "ORT CPU device ID must be 0 but got ", device.Id());
    return c10::Device(c10::DeviceType::CPU);
  } else if(device.Type() == OrtDevice::GPU) {
    ORT_ENFORCE(device.Id() >= 0, "ORT GPU device ID must be >= 0 but got ", device.Id());
    return c10::Device(c10::DeviceType::CUDA, device.Id());
  } else {
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

// Extract tensor's shape from onnxruntime::TensorShape as a vector.
std::vector<int64_t> create_shape_vector(const onnxruntime::TensorShape& shape) {
  const auto num_dims = shape.NumDimensions();
  std::vector<int64_t> shape_vec(num_dims);
  shape.CopyDims(shape_vec.data(), num_dims);
  return shape_vec;
}

// Create at::Tensor from onnxruntime::Tensor and keep
// onnxruntime::Tensor alive until at::Tensor is dead.
c10::IValue create_c10_ivalue_tensor(OrtValue value) {
  onnxruntime::Tensor* tensor = value.GetMutable<onnxruntime::Tensor>();
  const OrtDevice& device = tensor->Location().device;
  auto options = torch::TensorOptions()
    .dtype(create_torch_element_type(tensor->DataType()->AsPrimitiveDataType()))
    .layout(torch::kStrided)
    .device(create_c10_device(device))
    .requires_grad(false);

  std::vector<int64_t> shape = create_shape_vector(tensor->Shape());

  at::Tensor new_tensor = torch::from_blob(
    tensor->MutableDataRaw(),
    shape,
    // Capture-by-value means
    //  1. A new OrtValue is initialized from "value".
    //  2. The new OrtValue and "value" share the same underlying tensor, so
    //     the tensor's lifetime is controlled by both of them, whichever is longer.
    //  3. The new OrtValue's lifetime is the same as this lambda function.
    //  4. This lambda function is deleted by "new_tensor"'s dtor, which also ends
    //     the underlying tensor's life.
    [value] (void*) { },
    options);

  return c10::IValue(new_tensor);   
}

bool Accelerator::supported(const torch::jit::Node* node) {
  switch (node->kind()) {
    //case aten::relu:
    case aten::mul:
    case aten::add:
    case aten::sub:
    case aten::div:
    //case aten::gt:
    //case aten::eq:
    case prim::Constant:
    //case aten::threshold_backward:
      std::cout << "[compiler.cc] Support " << *node;  //<< std::endl;
      return true;
    default:
      std::cout << "[compiler.cc] Not support " << *node; //<< std::endl;
      return false;
  }
}

void Accelerator::run(torch::jit::Stack& stack) {
  // Get the number of expected inputs to the graph we are compiling
  const at::ArrayRef<torch::jit::Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();

  // Pop these inputs from the stack.
  at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, num_inputs);

  std::cout << "JIT sub-graph: " << std::endl;
  std::cout << *subgraph_ << std::endl;
  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  torch::jit::CompleteArgumentSpec spec{false, at::ArrayRef<c10::IValue>(inputs)};
  if (cache_.find(spec) == cache_.end()) {
    cache_[spec] = compile(spec, inputs);
  }

  // Run the compiled function!
  auto outputs = cache_[spec](inputs);

  torch::jit::drop(stack, num_inputs);

  for (auto& output : outputs) {
    auto var = torch::autograd::make_variable(output.toTensor());
    stack.push_back(c10::IValue(var));
  }
}

void Accelerator::check_inputs(
  const at::ArrayRef<c10::IValue>& inputs) {
  // TODO: remove this check.
  TORCH_CHECK(inputs.size(), "Need at least one input.");
  for (const auto& input : inputs) {
    TORCH_CHECK(input.isTensor(), "Compiler can only handle Tensor inputs.");
  }
}

// Store input types in sub-graph so that
// ONNX exporter can use them. Input types
// are required when executing ONNX model
// in ORT.
// TODO: Allow ORT to accept models without
// input types. Then, we can remove this function.
void Accelerator::propagate_input_types(
  const at::ArrayRef<c10::IValue>& inputs) {
  TORCH_CHECK(subgraph_->inputs().size() == inputs.size(),
    "Number of provided inputs must match captured sub-graph's schema.");
  const auto num_inputs = subgraph_->inputs().size();
  for (size_t i = 0; i < num_inputs; ++i) {
      auto input_symbol = subgraph_->inputs()[i];
      auto input_value = inputs[i];
      input_symbol->setType(input_value.type());
  }
}

// ONNX exporter is written in Python, so
// this function may calls some Python functions.
// Be aware of GIL issue.
// The returned value is the path to exported
// ONNX file.
std::string Accelerator::export_to_onnx() {
  pybind11::gil_scoped_acquire guard{};
  // Retrieve Python function.
  pybind11::function to_onnx =
      pybind11::reinterpret_borrow<pybind11::function>(
          pybind11::module::import("torch.onnx.utils").attr("_optimize_graph_1")
      );
  // Execute Python function.
  auto result = to_onnx(subgraph_, ::torch::onnx::OperatorExportTypes::ONNX);
  return result.cast<std::string>();
}

std::unique_ptr<onnxruntime::training::TrainingSession> Accelerator::create_session() {
  static onnxruntime::Environment& pybind_default_env = GetLtcEnv();
  static onnxruntime::SessionOptions sess_opts;
  return std::make_unique<onnxruntime::training::TrainingSession>(sess_opts, pybind_default_env);
}


static OrtDevice CheckAndGetTensorDevice(at::ArrayRef<c10::IValue>& values) {
  // This memory info must be shared by all tensors;
  // for example, all tensors on CPU or all on a specific GPU.
  // When all values are not tensors, we assume CPU device.
  // c10::Device's index is default to -1.
  c10::Device unique_tensor_device(c10::DeviceType::CPU);
  bool assigned = false;
  for (auto value : values) {
    if (!value.isTensor()) {
      continue;
    }
    auto tensor = value.toTensor();
    if (assigned) {
      // A device has been recorded, so we compare
      // it with the current tensor's device. 
      TORCH_CHECK(unique_tensor_device == tensor.device(),
        "All tensors must be on the same device.");
    } else {
      // Record the 1st tensor device.
      unique_tensor_device = tensor.device();
      assigned = true;
    }
  }
  return create_ort_device(unique_tensor_device);
}

CompiledCode Accelerator::compile(
    torch::jit::CompleteArgumentSpec spec, at::ArrayRef<c10::IValue>& inputs) {
  check_inputs(inputs);
  propagate_input_types(inputs);
  std::string model_path = export_to_onnx();
  cached_sess_.emplace(spec, create_session());
  onnxruntime::training::TrainingSession& sess = *cached_sess_.at(spec);

  OrtCUDAProviderOptions provider_options{};
  provider_options.do_copy_in_default_stream = true;
  auto factory = onnxruntime::CreateExecutionProviderFactory_Cuda(&provider_options);
  ORT_THROW_IF_ERROR(sess.RegisterExecutionProvider(factory->CreateProvider()));

  ORT_THROW_IF_ERROR(sess.Load(model_path));
  ORT_THROW_IF_ERROR(sess.Initialize());

  onnxruntime::RunOptions run_options;
  std::vector<std::string> feed_names;
  std::vector<std::string> fetch_names;

  for (auto node_arg : *sess.GetModelInputs().second) {
    feed_names.push_back(node_arg->Name());
  }
  for (auto node_arg : *sess.GetModelOutputs().second) {
    fetch_names.push_back(node_arg->Name());
  }

  // Memory info for all tensors.
  OrtDevice shared_device = CheckAndGetTensorDevice(inputs);
  std::vector<OrtDevice> fetches_device_info(fetch_names.size(), shared_device);

  // This function wraps the function pointer we bound our assembly to
  // Adheres to the CompiledCode interface defined in compiler.h
  auto compiled_func = [this, spec, run_options, feed_names, fetch_names, fetches_device_info](at::ArrayRef<c10::IValue>& inputs) {
    onnxruntime::training::TrainingSession& sess = *cached_sess_.at(spec);
    std::vector<OrtValue> feeds;
    std::vector<OrtValue> fetches;

    // LazyTensor backend assumes all tensors are on the same device.
    OrtMemoryInfo tensor_memory_info;
    const auto num_inputs = subgraph_->inputs().size();
    for (size_t i = 0; i < num_inputs; ++i) {
        if (subgraph_->inputs().at(i)->type()->kind() == c10::TypeKind::TensorType) {
          feeds.push_back(create_ort_tensor_value(inputs.at(i).toTensor()));
        } else {
          // TODO: handle other type correctly.
          ORT_THROW("Only tensor inputs are supported.");
          //feeds.push_back(create_ort_scalar_value(inputs.at(i).toScalar()));
        }
    }

    std::cout << "[accelerator.cpp] sess.Run" << std::endl;
    ORT_THROW_IF_ERROR(sess.Run(run_options, feed_names, feeds, fetch_names, &fetches, &fetches_device_info));
    std::cout << "[accelerator.cpp] sess.Run done" << std::endl;

    std::vector<c10::IValue> outputs;
    for (auto value : fetches) {
        outputs.push_back(std::move(create_c10_ivalue_tensor(value)));
    }

    return outputs;
  };

  return compiled_func;
}
