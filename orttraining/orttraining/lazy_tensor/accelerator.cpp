// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "accelerator.h"
#include "flags.h"
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
#include <sstream>

namespace onnxruntime {
namespace lazytensor {

namespace py = pybind11;
namespace aten = torch::jit::aten;
namespace prim = torch::jit::prim;

// static variable used to create inference session and training session.
const static std::string env_name = std::string("LTC");
static std::unique_ptr<onnxruntime::Environment> ltc_env;

std::string ToString(const c10::IValue& value) {
  std::stringstream ss;
  if (value.isTensor()) {
    // Produce, e.g., Tensor<Float>(1024, 128)@cpu.
    const auto& tensor = value.toTensor();
    ss << "Tensor"
       << "<" << c10::toString(tensor.scalar_type()) << ">";
    if (tensor.sizes().empty()) {
    } else {
      ss << "(";
      for (int i = 0; i < tensor.dim(); i++) {
        ss << tensor.sizes()[i];
        if (i != tensor.dim() - 1) {
          ss << ",";
        }
      }
      ss << ")";
    }
    ss << "@" << tensor.device();
  } else if (value.isScalar()) {
    // Produce, e.g., Scalar<Float>, which is always on CPU.
    ss << "Scalar<" << c10::toString(value.toScalar().type()) << ">";
  } else {
    ORT_THROW("Unsupported type.");
  }
  return ss.str();
}

bool CompareTensor(const at::Tensor& left, const at::Tensor& right) {
  if (left.sizes() != right.sizes()) {
    return false;
  }
  if (left.scalar_type() != right.scalar_type()) {
    return false;
  }
  if (left.device() != right.device()) {
    return false;
  }
  // Uncomment the following line to compare the content of the tensors.
  // if (!at::allclose(left, right)) {
  //   return false;
  // }
  return true;
}

bool CompareScalar(const at::Scalar& left, const at::Scalar& right) {
  if (left.type() == right.type()) {
    return true;
  } else {
    return false;
  }
}

bool Compare(const c10::IValue& left, const c10::IValue& right) {
  if (left.isTensor() && right.isTensor()) {
    return CompareTensor(left.toTensor(), right.toTensor());
  } else if (left.isScalar() && right.isScalar()) {
    return CompareScalar(left.toScalar(), right.toScalar());
  } else {
    return false;
  }
}

bool CompareStack(const torch::jit::Stack& left, const torch::jit::Stack& right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t i = 0; i < left.size(); i++) {
    if (!Compare(left[i], right[i])) {
      return false;
    }
  }
  return true;
}

// Print last n elements in the stack.
std::string ToString(const at::ArrayRef<c10::IValue>& values) {
  std::stringstream ss;
  for (size_t i = 0; i < values.size(); i++) {
    ss << ToString(values.at(i));
    if (i != values.size() - 1) {
      ss << ", ";
    }
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
    // TODO: add as many ops as possible.
    case aten::add:
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

void Accelerator::OrtRun(torch::jit::Stack& stack) {
  if (DumpGraph()) {
    std::cout << "[ORT,Graph]\n"
              << *subgraph_;
  }

  // Get these inputs from the stack.
  at::ArrayRef<c10::IValue> inputs = torch::jit::last(stack, subgraph_->inputs().size());
  // If we haven't compiled for the shape/device of these inputs before,
  // do so now.
  // Compile a callable to execute "subgraph_" on the inputs.
  // If such input schema appears before, we can reuse a cached compiled callable.
  torch::jit::CompleteArgumentSpec spec{false, inputs};
  if (cache_.find(spec) == cache_.end()) {
    cache_.emplace(spec, Compile(spec, inputs));
  }

  if (DumpInputsOutputs()) {
    std::cout << "[ORT,Input] " << ToString(inputs) << std::endl;
    ;
  }

  // Run the compiled function!
  auto outputs = cache_[spec].code(inputs);

  // Discard used inputs.
  torch::jit::drop(stack, inputs.size());

  // Return results to caller.
  for (auto& output : outputs) {
    stack.push_back(output);
  }

  if (DumpInputsOutputs()) {
    at::ArrayRef<c10::IValue> outputs = torch::jit::last(stack, subgraph_->outputs().size());
    std::cout << "[ORT,Output] " << ToString(outputs) << std::endl;
  }
}

void Accelerator::PytorchRun(torch::jit::Stack& stack) {
  if (DumpGraph()) {
    std::cout << "[Pytorch,Graph]\n"
              << *subgraph_;
  }
  if (DumpInputsOutputs()) {
    at::ArrayRef<c10::IValue> inputs = torch::jit::last(
        stack, subgraph_->inputs().size());
    std::cout << "[PyTorch,Input] " << ToString(inputs) << std::endl;
  }

  torch::jit::GraphExecutor executor(subgraph_, "");
  executor.run(stack);

  if (DumpInputsOutputs()) {
    at::ArrayRef<c10::IValue> outputs = torch::jit::last(
        stack, subgraph_->outputs().size());
    std::cout << "[PyTorch,Output] " << ToString(outputs) << std::endl;
  }
}

void Accelerator::DebugRun(torch::jit::Stack& stack) {
  torch::jit::Stack copy;
  copy = stack;
  OrtRun(stack);
  PytorchRun(copy);
  ORT_ENFORCE(CompareStack(stack, copy),
              "ORT and Pytorch must generate the same results \
              but tensor types or shapes are different.");
}

void Accelerator::Run(torch::jit::Stack& stack) {
  if (CheckBaseline()) {
    // Run both ORT and Pytorch to execute the subgraph
    // and compare their output types and shapes.
    DebugRun(stack);
  } else {
    OrtRun(stack);
  }
}

static void CheckArgs(
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
static void PropagateArgTypes(
    const at::ArrayRef<c10::IValue>& inputs,
    std::shared_ptr<torch::jit::Graph> graph) {
  TORCH_CHECK(graph->inputs().size() == inputs.size(),
              "Number of provided inputs must match captured sub-graph's schema.");
  for (size_t i = 0; i < graph->inputs().size(); ++i) {
    auto input_symbol = graph->inputs()[i];
    auto input_value = inputs[i];
    if (!input_value.isTensor()) {
      // The allowed IR components in ONNX exporter and Pytorch
      // are a little different. I am not confident to fill
      // types other than tensor, because of the ambiguous scalar
      // representations in Pytorch.
      continue;
    }
    input_symbol->setType(input_value.type());
  }
}

// ONNX exporter is written in Python, so
// this function may calls some Python functions.
// Be aware of GIL issue.
// The returned value is the path to exported
// ONNX file.
static std::string ExportToOnnx(
  std::shared_ptr<torch::jit::Graph> graph,
    const at::ArrayRef<c10::IValue>& args
) {
  // ONNX exporter modifies the graph in-place, so we
  // need to clone it to avoid interaction between
  // Pytorch's JIT mechanism and ONNX graph.
  std::shared_ptr<torch::jit::Graph> new_subgraph(graph->copyUnique().release());
  // Acquire GIL since Python is not multi-threading.
  pybind11::gil_scoped_acquire guard{};
  // Retrieve Python exporter function.
  pybind11::function export_to_onnx =
      pybind11::reinterpret_borrow<pybind11::function>(
          pybind11::module::import("torch.onnx.utils").attr("_optimize_graph_1"));
  // Execute Python function.

  PropagateArgTypes(args, new_subgraph);
  auto result = export_to_onnx(new_subgraph, ::torch::onnx::OperatorExportTypes::ONNX);
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

// Initialize empty session with ONNX model.
static void InitializeSession(
  const std::string& model_path, onnxruntime::InferenceSession& sess) {
  // Add EPs.
#ifdef USE_CUDA
  // When CUDA is enabled, some CUDA-only graph graph fusions are enabled.
  // If we don't add CUDA EP, ONNX Runtime may throw even when running MNIST.
  OrtCUDAProviderOptions provider_options{};
  provider_options.do_copy_in_default_stream = true;
  auto factory = onnxruntime::CreateExecutionProviderFactory_Cuda(&provider_options);
  ORT_THROW_IF_ERROR(sess.RegisterExecutionProvider(factory->CreateProvider()));
#endif
  ORT_THROW_IF_ERROR(sess.Load(model_path));
  ORT_THROW_IF_ERROR(sess.Initialize());
}

CompiledObject Accelerator::Compile(
    torch::jit::CompleteArgumentSpec spec, at::ArrayRef<c10::IValue>& args) {
  CheckArgs(args);
  // Storage of compilation.
  CompiledObject compiled;
  // Assign an empty session.
  compiled.sess = CreateSession();
  // Let's get the empty session and initialize it.
  onnxruntime::InferenceSession& sess = *compiled.sess;
  // Export subgraph_ to ONNX.
  const std::string model_path = ExportToOnnx(subgraph_, args);
  // Load ONNX model into session, register
  // EPs and finally initialize session.
  InitializeSession(model_path, sess);
  // Clean model file.
  ORT_ENFORCE(std::remove(model_path.c_str()) == 0, "Failed to remove temporary file: ", model_path);

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

  // Create a callable which feeds inputs to ORT
  // session's Run(...) and returns outputs.
  auto code = [this, run_options,
               feed_names, fetch_names,
               fetches_device_info, &sess](at::ArrayRef<c10::IValue>& args) {
    // Inputs of ORT session.
    std::vector<OrtValue> feeds;
    // Outputs of ORT session.
    std::vector<OrtValue> fetches;

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

    // Inputs are ready. Let's run ORT.
    ORT_THROW_IF_ERROR(sess.Run(
        run_options,
        feed_names, feeds,
        fetch_names, &fetches, &fetches_device_info));

    // Convert ORT output to Pytorch format.
    std::vector<c10::IValue> outputs;
    for (auto value : fetches) {
      if (value.IsTensor()) {
        outputs.push_back(std::move(CreateC10IvalueTensor(value)));
      } else {
        ORT_ENFORCE("ORT must return tensors.");
      }
    }

    return outputs;
  };

  compiled.code = code;
  return compiled;
}
}  // namespace lazytensor
}  // namespace onnxruntime
