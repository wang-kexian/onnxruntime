// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "accelerator.h"
#include <iostream>
#include <string>
#include <torch/csrc/jit/passes/onnx.h>
#include "torch/csrc/jit/passes/shape_analysis.h"
#include <torch/torch.h>
#include "bridge.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/session_options.h"
#include "core/session/environment.h"
#include "python/onnxruntime_pybind_state_common.h"
#include <cstdlib>     /* getenv */
#include <sstream>

namespace onnxruntime {
namespace lazytensor {

namespace py = pybind11;
namespace aten = torch::jit::aten;
namespace prim = torch::jit::prim;

// static variable used to create inference session and training session.
const static std::string env_name = std::string("LTC");
static std::unique_ptr<onnxruntime::Environment> ltc_env;

std::string ToString(c10::IValue& value) {
  std::stringstream ss;
  if (value.isTensor()) {
    auto& tensor = value.toTensor();
    ss << "Tensor<" << c10::toString(tensor.scalar_type()) << ">";
    if (tensor.sizes().empty()) {
      ss << "()";
    } else {
      ss << "(";
      for (int i = 0; i < tensor.dim(); i++) {
        ss << tensor.sizes()[i];
        if (i != tensor.dim() - 1) {
          ss << ", ";
        }
      }
      ss << ")";
    }
  } else if (value.isScalar()) {
    ss << "Scalar<" << c10::toString(value.toScalar().type()) << ">";
  }
  return ss.str();
}

onnxruntime::Environment& GetLtcEnv() {
  if (!ltc_env) {
    ORT_THROW_IF_ERROR(
        onnxruntime::Environment::Create(
            std::make_unique<onnxruntime::logging::LoggingManager>(
                std::make_unique<onnxruntime::logging::CLogSink>(),
                onnxruntime::logging::Severity::kWARNING,
                false,
                onnxruntime::logging::LoggingManager::InstanceType::Temporal,
                &env_name),
            ltc_env));
  }
  return *ltc_env;
}

bool Accelerator::Supported(const torch::jit::Node* node) {
  if (!node) {
    return false;
  }

  switch (node->kind()) {
    case aten::relu:
    case aten::mul:
    case aten::sub:
    case aten::div:
    case aten::gt:
    case aten::lt:
    case aten::eq:
    case prim::Constant:
    case aten::sqrt:
    case aten::permute:
    case aten::mm:
    case aten::ne:
    case aten::abs:
    case aten::max:
    case aten::min:
      return true;
    default:
      return false;
  }
}

void Accelerator::PthRun(torch::jit::Stack& stack) {
  const at::ArrayRef<torch::jit::Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();
  const at::ArrayRef<torch::jit::Value*>& graph_outputs = subgraph_->outputs();
  const auto num_outputs = graph_outputs.size();

  at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, num_inputs);
  for (auto input: inputs) {
    std::cout << "pth-input: " << ToString(input) << std::endl;
  }

  torch::jit::GraphExecutor graph_exec_(subgraph_, "");
  graph_exec_.run(stack);

  at::ArrayRef<c10::IValue> outputs = torch::jit::last(stack, num_outputs);
  for (auto output: outputs) {
    std::cout << "pth-output: " << ToString(output) << std::endl;
  }
}

void Accelerator::Run(torch::jit::Stack& stack) {
  std::cout << "---------------------ORT-v.s.-PTH---------------------" << std::endl;
  char* flag = NULL;
  flag = std::getenv("USEORT");
  char* print_jit_graph = NULL;
  print_jit_graph = std::getenv("PRINTJITGRAPH");
  if (print_jit_graph != NULL) {
    std::cout << "ORT-Received sub-graph:\n" << *subgraph_ << std::endl;
  }
  if (flag != NULL) {
    torch::jit::Stack& ort_stack = stack;
    OrtRun(ort_stack);
  } else {
    torch::jit::Stack& torch_stack = stack;
    PthRun(torch_stack);
  }
  std::cout << "---------------------ORT-v.s.-PTH done-----------------" << std::endl;
}

void Accelerator::OrtRun(torch::jit::Stack& stack) {
  // Get the number of expected inputs to the graph we are compiling
  const at::ArrayRef<torch::jit::Value*>& graph_inputs = subgraph_->inputs();
  const auto num_inputs = graph_inputs.size();
  //const at::ArrayRef<torch::jit::Value*>& graph_outputs = subgraph_->outputs();
  //const auto num_outputs = graph_outputs.size();

  // Pop these inputs from the stack.
  at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, num_inputs);
  //int i = 0;
  for (auto input: inputs) {
    std::cout << "ort-input: " << ToString(input) << std::endl;
  }

  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  torch::jit::CompleteArgumentSpec spec{false, at::ArrayRef<c10::IValue>(inputs)};
  if (cache_.find(spec) == cache_.end()) {
    cache_.emplace(spec, Compile(spec, inputs));
  }

  // Run the compiled function!
  auto outputs = cache_[spec].code(inputs);
  //i = 0;
  for (auto output: outputs) {
    std::cout << "ort-output: " << ToString(output) << std::endl;
  }

  torch::jit::drop(stack, num_inputs);

  for (auto& output : outputs) {
    stack.push_back(output);
  }
}

void Accelerator::CheckArgs(
    const at::ArrayRef<c10::IValue>& inputs) {
  // TODO: remove this check.
  TORCH_CHECK(inputs.size(), "Need at least one input.");
  for (const auto& input : inputs) {
    TORCH_CHECK(input.isTensor() || input.isScalar(), "Compiler can only handle Tensor or Scalar inputs.");
  }
}

// Store input types in sub-graph so that
// ONNX exporter can use them. Input types
// are required when executing ONNX model
// in ORT.
// TODO: Allow ORT to accept models without
// input types. Then, we can remove this function.
void Accelerator::PropagateArgTypes(
    const at::ArrayRef<c10::IValue>& inputs) {
  TORCH_CHECK(subgraph_->inputs().size() == inputs.size(),
              "Number of provided inputs must match captured sub-graph's schema.");
  const auto num_inputs = subgraph_->inputs().size();
  for (size_t i = 0; i < num_inputs; ++i) {
    auto input_symbol = subgraph_->inputs()[i];
    auto input_value = inputs[i];
    if (input_value.isTensor()) {
      input_symbol->setType(input_value.type());
    }
  }
  //std::cout << "JIT sub-graph: " << std::endl;
  //std::cout << *subgraph_ << std::endl;
  torch::jit::PropagateInputShapes(subgraph_);
  //std::cout << "JIT sub-graph with shpaes: " << std::endl;
  //std::cout << *subgraph_ << std::endl;
}

// ONNX exporter is written in Python, so
// this function may calls some Python functions.
// Be aware of GIL issue.
// The returned value is the path to exported
// ONNX file.
static std::string ExportToOnnx(std::shared_ptr<torch::jit::Graph> graph) {
  pybind11::gil_scoped_acquire guard{};
  // Retrieve Python exporter function.
  pybind11::function export_to_onnx =
      pybind11::reinterpret_borrow<pybind11::function>(
          pybind11::module::import("torch.onnx.utils").attr("_optimize_graph_1"));
  // Execute Python function.
  auto result = export_to_onnx(graph, ::torch::onnx::OperatorExportTypes::ONNX);
  return result.cast<std::string>();
}

// Create an empty session object.
// Models will be loaded later.
static std::unique_ptr<onnxruntime::InferenceSession> CreateSession() {
  // Enviroment shared by all sessions.
  static onnxruntime::Environment& pybind_default_env = GetLtcEnv();
  // All sessions use the same config.
  static onnxruntime::SessionOptions sess_opts;
  return std::make_unique<onnxruntime::InferenceSession>(sess_opts, pybind_default_env);
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
  return CreateOrtDevice(unique_tensor_device);
}

/*
std::string GetC10TypeString(c10::IValue& value) {

}
*/

CompiledObject Accelerator::Compile(
    torch::jit::CompleteArgumentSpec spec, at::ArrayRef<c10::IValue>& args) {
  CompiledObject compiled;
  // Assign an empty session.
  compiled.sess = CreateSession();
  // Let's get the empty session and initialize it.
  onnxruntime::InferenceSession& sess = *compiled.sess;

  OrtCUDAProviderOptions provider_options{};
  provider_options.do_copy_in_default_stream = true;
  auto factory = onnxruntime::CreateExecutionProviderFactory_Cuda(&provider_options);
  ORT_THROW_IF_ERROR(sess.RegisterExecutionProvider(factory->CreateProvider()));

  // Export from Pytorch and load ONNX model into session.
  CheckArgs(args);
  PropagateArgTypes(args);
  std::string model_path = ExportToOnnx(subgraph_);
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
  // Assume all inputs and outputs are on the same device.
  OrtDevice shared_device = CheckAndGetTensorDevice(args);
  // Duplicate device info for each output tensor.
  // TODO: Force scalar to be on CPU since at::Scalar is CPU value.
  std::vector<OrtDevice> fetches_device_info(fetch_names.size(), shared_device);

  auto code = [this, spec, run_options,
               feed_names, fetch_names,
               fetches_device_info, &sess, model_path](at::ArrayRef<c10::IValue>& args) {
    // Inputs of ORT session.
    std::vector<OrtValue> feeds;
    // Outputs of ORT session.
    std::vector<OrtValue> fetches;
    std::cout << "Execute ONNX model " << model_path << std::endl;

    // Prepare inputs.
    const auto num_inputs = subgraph_->inputs().size();
    for (size_t i = 0; i < num_inputs; ++i) {
      // The value can be either tensor or scalar.
      // Scalar is a tensor with empty shape vector.
      // Create ORT tensor from Pytorch tensor without copy.
      if (args.at(i).isScalar()) {
        // Scalar.
        // ORT_ENFORCE(subgraph_->inputs().at(i)->type()->kind() == c10::TypeKind::TensorType);
        feeds.push_back(CreateOrtScalarValue(args.at(i).toScalar()));
      } else if (args.at(i).isTensor()) {
        // Tensor.
        ORT_ENFORCE(subgraph_->inputs().at(i)->type()->kind() == c10::TypeKind::TensorType);
        feeds.push_back(CreateOrtTensorValue(args.at(i).toTensor()));
      } else {
        // Looks like LTC only passes scalars and tensors into backend, so we don't care
        // other types for now.
        ORT_THROW("Only tensor inputs are supported.");
      } 
    }

    std::cout << "Run" << std::endl;
    // Inputs are ready. Let's run ORT.
    ORT_THROW_IF_ERROR(sess.Run(
        run_options,
        feed_names, feeds,
        fetch_names, &fetches, &fetches_device_info));
    std::cout << "Run done" << std::endl;

    // Convert ORT output to Pytorch format.
    std::vector<c10::IValue> outputs;
    for (auto value : fetches) {
      if (value.IsTensor()) {
        onnxruntime::Tensor* tensor = value.GetMutable<onnxruntime::Tensor>();
        const onnxruntime::TensorShape& tensor_shape = tensor->Shape();
        if (tensor_shape.NumDimensions() > 0) {
          // Create Pytorch tensor from ORT tensor without copy.
          outputs.push_back(std::move(CreateC10IvalueTensor(value)));
        } else if (tensor_shape.NumDimensions() == 0) {
          outputs.push_back(std::move(CreateC10IvalueScalar(value)));
        } else {
          ORT_ENFORCE("Unsupported tensor shape.");
        }
      } else {
        ORT_ENFORCE("Output must be tensor or scalar.");
      }
    }

    std::cout << "Execute ONNX model done" << model_path << std::endl;
    return outputs;
  };

  compiled.code = code;
  return compiled;
}
}  // namespace lazytensor
}  // namespace onnxruntime
