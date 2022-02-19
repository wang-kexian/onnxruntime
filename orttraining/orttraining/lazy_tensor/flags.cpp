// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flags.h"
#include <cstdlib>
#include <cstring>
#include "core/common/common.h"

namespace onnxruntime {
namespace lazytensor {
bool IsEnvironmentVariableOne(const char* name) {
  const auto flag = std::getenv(name);
  if (flag == nullptr) {
    return false;
  }
  const auto is_one = std::strcmp(flag, "1") == 0;
  const auto is_zero = std::strcmp(flag, "0") == 0;
  ORT_ENFORCE(is_one || is_zero,
              "Must set ", name, "=0, ", name, "=1, or unset ", name);
  return is_one;
}

// If returned value is true, run torch::jit::GraphExecutor
// and compare its outputs with ORT's outputs.
// Only types and shapes are compared.
bool CheckBaseline() {
  return IsEnvironmentVariableOne("ORTLTCHECKBASELINE");
}

// When returing true, we dump the inputs and outputs
// when ORT (and Pytorch when ORTLTCHECKBASELINE is set to 1)
// executes the subgraph.
bool DumpInputsOutputs() {
  return IsEnvironmentVariableOne("ORTLTDUMPINPUTSOUTPUTS");
}

// Returns true to dump the torch::jit::Graph ORT receives
// from LazyTensor.
bool DumpGraph() {
  return IsEnvironmentVariableOne("ORTLTDUMPGRAPH");
}
}  // namespace lazytensor
}  // namespace onnxruntime