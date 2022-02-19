// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace lazytensor {
// If returned value is true, run torch::jit::GraphExecutor
// and compare its outputs with ORT's outputs.
// Only types and shapes are compared.
bool CheckBaseline();
// When returing true, we dump the inputs and outputs
// when ORT (and Pytorch when ORTLTCHECKBASELINE is set to 1)
// executes the subgraph.
bool DumpInputsOutputs();
// Returns true to dump the torch::jit::Graph ORT receives
// from LazyTensor.
bool DumpGraph();
}  // namespace lazytensor
}  // namespace onnxruntime