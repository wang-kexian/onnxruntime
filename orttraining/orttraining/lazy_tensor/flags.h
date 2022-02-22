// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace lazytensor {
// This file contains environment variables that control
// the behavior of ORT as LazyTensor's backend.
// Most variables are for debug purpose.

// When returing true, we dump the inputs and outputs
// when ORT (and Pytorch when ORTLTCHECKBASELINE is set to 1)
// executes the subgraph.
bool DumpInputsOutputs();
// Returns true to dump the torch::jit::Graph ORT receives
// from LazyTensor.
bool DumpGraph();
// If returned value is true, run torch::jit::GraphExecutor
// and compare its outputs with ORT's outputs.
// Only types and shapes are compared. The user can control
// the checking mechanism. For example, set
// ORT_LT_CHECK_TENSOR_CONTENT=1 to compare tensor elements.
//
// Related functions' dependency graph:
//  CheckBaseline -> CheckTensorContent -> AbsoluteTolerance
//                                   '---> RelativeTolerance
bool CheckBaseline();
// If this function returns true, check tensor's elements
// when CheckBaseline() returns true.
bool CheckTensorContent();
// The "absolute_tol" in
// |value-expected| <= |expected| * relative_tol + absolute_tol
double AbsoluteTolerance();
// The "relative_tol" in
// |value-expected| <= |expected| * relative_tol + absolute_tol
double RelativeTolerance();
}  // namespace lazytensor
}  // namespace onnxruntime