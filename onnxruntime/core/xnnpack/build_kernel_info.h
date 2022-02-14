#pragma once

namespace onnxruntime {
namespace xnnpack {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}
}  // namespace onnxruntime