#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include "core/session/onnxruntime_cxx_api.h"
#include "orttraining/core/session/training_session.h"

using CompiledCode = std::function<std::vector<c10::IValue>(
    at::ArrayRef<c10::IValue>&)>;

class Accelerator {
 public:
  Accelerator(const torch::jit::Node* node)
      : subgraph_(node->g(torch::jit::attr::Subgraph)) { std::cout << "JIT see\n" << *node << std::endl; }
  void Run(torch::jit::Stack& stack);
  static bool Supported(const torch::jit::Node* node);

 private:

  void CheckArgs(const at::ArrayRef<c10::IValue>& args);
  void PropagateArgTypes(const at::ArrayRef<c10::IValue>& args);
  CompiledCode Compile(
    torch::jit::CompleteArgumentSpec spec, at::ArrayRef<c10::IValue>& args);
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, CompiledCode> cache_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, std::unique_ptr<onnxruntime::training::TrainingSession>> cached_sess_;
};
