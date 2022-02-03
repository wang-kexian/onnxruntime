// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include "core/session/onnxruntime_cxx_api.h"
#include "orttraining/core/session/training_session.h"

namespace onnxruntime {
namespace lazytensor {

// Type of JIT compilation result.
struct CompiledObject {
  // Callable to execute the computation represented by torch::jit::Graph.
  // It processes tensors across ORT and Pytorch and invokes "sess".
  std::function<std::vector<c10::IValue>(at::ArrayRef<c10::IValue>&)> code;
  // Session used in the "code" above.
  std::unique_ptr<onnxruntime::training::TrainingSession> sess;
};

// Custom JIT engine called by Pytorch.
class Accelerator {
 public:
  Accelerator(const torch::jit::Node* node)
      : subgraph_(node->g(torch::jit::attr::Subgraph)) {}
  // Execute a call to the torch::jit::Graph represented by "subgraph_".
  // This function could compile the graph and cache the result
  // for repeated uses.
  void Run(torch::jit::Stack& stack);
  // Determine if this node can be translated to ONNX.
  static bool Supported(const torch::jit::Node* node);

 private:
  void CheckArgs(const at::ArrayRef<c10::IValue>& args);
  void PropagateArgTypes(const at::ArrayRef<c10::IValue>& args);
  CompiledObject Compile(
      torch::jit::CompleteArgumentSpec spec, at::ArrayRef<c10::IValue>& args);
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, CompiledObject> cache_;
};
}  // namespace lazytensor
}  // namespace onnxruntime
