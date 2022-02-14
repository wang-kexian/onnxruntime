// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The program convert a model from NCHW format to NHWC format. It keeps conv weights unchanged.

#include <core/session/onnxruntime_c_api.h>
#include <core/graph/model.h>
#include <core/common/common.h>
#include <core/graph/model.h>
#include <core/platform/logging/make_platform_default_log_sink.h>
#include <onnx/defs/operator_sets.h>
#include <core/graph/contrib_ops/ms_opset.h>
#include <core/graph/contrib_ops/onnx_deprecated_opset.h>
#include <core/optimizer/transpose_optimizer/api_impl.h>
#include <core/optimizer/transpose_optimizer/api.h>
#include <onnx/shape_inference/implementation.h>
#include <core/xnnpack/schema/xnnpack_opset.h>
#include <core/framework/op_node_proto_helper.h>
#include <core/providers/common.h>
#include <core/optimizer/selectors_actions/helpers.h>
#include <core/framework/tensorprotoutils.h>
#include "core/optimizer/utils.h"

using namespace onnxruntime;

#ifdef _WIN32
#define ORT_RETURN_NEG_ONE_IF_ERROR(expr)                              \
  do {                                                                 \
    auto _status = (expr);                                             \
    if ((!_status.IsOK())) {                                           \
      std::wcout << ToWideString(_status.ErrorMessage()) << std::endl; \
      return -1;                                                       \
    }                                                                  \
  } while (0)
#else
#define ORT_RETURN_NEG_ONE_IF_ERROR(expr)               \
  do {                                                  \
    auto _status = (expr);                              \
    if ((!_status.IsOK())) {                            \
      std::cout << _status.ErrorMessage() << std::endl; \
      return -1;                                        \
    }                                                   \
  } while (0)
#endif

void foo(Graph& graph, std::shared_ptr<CPUAllocator>& cpu_allocator) {
  auto api_graph = MakeApiGraph(graph, cpu_allocator, kCpuExecutionProvider);
  bool modified = false;
  for (std::unique_ptr<onnx_layout_transformation::api::NodeRef>& node : api_graph->Nodes()) {
    // Only QLinearConv needs to be handled explicitly. The rest will be transformed if needed during transpose
    // optimization.
    if (node->OpType() == "Conv") {
      auto domain = node->Domain();

      // Skip if domain is incorrect
      if (domain != kOnnxDomain && domain != kOnnxDomainAlias) {
        continue;
      }

      // Skip if unknown rank
      auto shape = NodeFromApiNode(*node).InputDefs()[0]->Shape();
      if (shape == nullptr) {
        continue;
      }

      // Convert to channels last
      size_t rank = shape->dim_size();
      std::vector<int64_t> input_perm = onnx_layout_transformation::ChannelFirstToLastPerm(rank);
      std::vector<int64_t> output_perm = onnx_layout_transformation::ChannelLastToFirstPerm(rank);
      onnx_layout_transformation::WrapTransposesAroundNode(*api_graph, *node, {&input_perm}, {&output_perm});

      if (domain != kMSDomain) {
        auto inputs = node->Inputs();
        auto outputs = node->Outputs();
        auto new_node = api_graph->AddNode("NhwcConv", inputs, outputs.size(), kMSDomain, node->Name());
        for (size_t j = 0; j < outputs.size(); ++j) {
          if (outputs[j] != "") {
            api_graph->MoveOutput(*node, j, *new_node, j);
          }
        }
        new_node->CopyAttributes(*node);
        api_graph->RemoveNode(*node);
      }

      modified = true;
    }
  }

  if (modified) {
    onnx_layout_transformation::Optimize(*api_graph, /*allow_extended_ops*/ true);
  }
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int wmain(int argc, char* argv[]) {
#endif
  if (argc < 3) return -1;
  setlocale(LC_ALL, "");
  auto& domainToVersionRangeInstance = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
  if (domainToVersionRangeInstance.Map().find(onnxruntime::kMSDomain) == domainToVersionRangeInstance.Map().end()) {
    // External shared providers may have already added kMSDomain
    domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
  }
  domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSExperimentalDomain, 1, 1);
  domainToVersionRangeInstance.AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);
  domainToVersionRangeInstance.AddDomainToVersion("com.microsoft.xnnpack", 1, 1);

  ::ONNX_NAMESPACE::RegisterOnnxOperatorSetSchema();
  ::ONNX_NAMESPACE::RegisterOpSetSchema<contrib::OpSet_Microsoft_ver1>();
  ::ONNX_NAMESPACE::RegisterOpSetSchema<contrib::OpSet_ONNX_Deprecated>();
  ::ONNX_NAMESPACE::RegisterOpSetSchema<xnnpack::OpSet_XnnPack_ver1>();

  //::ONNX_NAMESPACE::RegisterOnnxMLOperatorSetSchema();
  Status status;
  std::string default_logger_id = "default";
  auto lmgr = std::make_unique<logging::LoggingManager>(logging::MakePlatformDefaultLogSink(),
                                                        logging::Severity::kINFO,
                                                        false,
                                                        logging::LoggingManager::InstanceType::Default,
                                                        &default_logger_id);

  ORT_ENFORCE(status.IsOK());
  const ORTCHAR_T* input_model_path = argv[1];
  const ORTCHAR_T* output_model_path = argv[2];
  auto logger = lmgr->CreateLogger("default");
  std::shared_ptr<onnxruntime::Model> m;
  ORT_ENFORCE(Model::Load(input_model_path, m, nullptr, *logger).IsOK());
  Graph& graph = m->MainGraph();
  GraphViewer graph_viewer(graph);
  std::shared_ptr<CPUAllocator> cpu_allocator = std::make_shared<CPUAllocator>();
  foo(graph, cpu_allocator);
  ORT_RETURN_NEG_ONE_IF_ERROR(m->MainGraph().Resolve());
  auto& main_graph = m->MainGraph();
  GraphViewer gv(main_graph);
  std::vector<NodeIndex> conv_nodes;
  for (auto& nodeRef : gv.Nodes()) {
    if (nodeRef.OpType() != "NhwcConv") continue;
    conv_nodes.push_back(nodeRef.Index());
  }
  for (NodeIndex ni : conv_nodes) {
    Node* node_p = graph.GetNode(ni);
    if (node_p == nullptr)
      continue;
    Node& nodeRef = *node_p;
    ProtoHelperNodeContext nc(nodeRef);
    int64_t group = 1;
    OpNodeProtoHelper info(&nc);
    auto X_input = info.GetInputType(0);
    auto weight_input = info.GetInputType(1);
    TensorShape weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_input->tensor_type().shape());
    TensorShape X_shape = utils::GetTensorShapeFromTensorShapeProto(X_input->tensor_type().shape());
    if (group != 1 && group != X_shape[3]) continue;
    //std::string auto_pad_str;
    //ORT_ENFORCE(info.GetAttr<std::string>("auto_pad", &auto_pad_str).IsOK());
    //AutoPadType auto_pad = StringToAutoPadType(auto_pad_str);
    ORT_ENFORCE(info.GetAttr<int64_t>("group", &group).IsOK());
    //group == 1 || group  == input / output channel count
    //For now we assume input channel count isn't 1, so that group count != input/output channel count
    bool is_depthwise = group == X_shape[3];
    //NodeArg* input = node.MutableInputDefs()[i];

    InOutDefSlot src_slot{ArgType::kInput, 1};
    InOutDefSlot dest_slot{ArgType::kInput, 0};
    //Append a single slot to dest. As the dest is empty, it will be the first one.
    ValueMoveInfo value_move_info(src_slot, ArgType::kInput, false, false);
    //const_cast
    Node* conv_node = graph.GetNode(nodeRef.Index());
    ORT_ENFORCE(conv_node != nullptr && conv_node->InputDefs().size() >= 3);
    for (size_t i = 0; i != weight_shape.NumDimensions(); ++i) {
      ORT_ENFORCE(weight_shape[i] > 0);
    }
    ORT_ENFORCE(weight_shape.NumDimensions() == 4);
    //bool is_pointwise = weight_shape[2] == 1 && weight_shape[3] == 1;
    //if (!is_pointwise) {
    std::vector<int64_t> input_perm = is_depthwise ? std::vector<int64_t>{1, 2, 3,0}  : std::vector<int64_t>{0, 2, 3, 1};
    std::string output_name = graph.GenerateNodeArgName("trans");
    NodeArg& transpose_output = graph.GetOrCreateNodeArg(output_name, nullptr);

    Node& dest_node = graph.AddNode("", "Transpose", "", {}, {&transpose_output}, nullptr, kOnnxDomain);
    dest_node.AddAttribute("perm", input_perm);
    ORT_ENFORCE(MoveInputOutput(graph, *conv_node, dest_node, value_move_info, false).IsOK());
    ORT_ENFORCE(graph.UpdateShapeInference(dest_node).IsOK());
    graph.AddEdge(dest_node.Index(), conv_node->Index(), 0, 1);

    //onnx_layout_transformation::TransposesNodeInputs(*api_graph, *nodeRef, 1, input_perm);
    std::vector<int64_t> strides, dilations;
    ORT_ENFORCE(info.GetAttrs<int64_t>("strides", strides).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("dilations", dilations).IsOK());

    //auto inputs = nodeRef->Inputs();
    //auto outputs = nodeRef->Outputs();
    std::string node_name = conv_node->Name();
    Node& new_node = graph.AddNode(node_name, is_depthwise?"XnnPackDepthwiseConvolution2d":"XnnPackConvolution2d", "", conv_node->MutableInputDefs(), {},
                                   nullptr, "com.microsoft.xnnpack");
    new_node.AddAttribute("input_padding_top", static_cast<int64_t>(0));
    new_node.AddAttribute("input_padding_right", static_cast<int64_t>(0));
    new_node.AddAttribute("input_padding_bottom", static_cast<int64_t>(0));
    new_node.AddAttribute("input_padding_left", static_cast<int64_t>(0));

    new_node.AddAttribute("subsampling_height", strides[0]);
    new_node.AddAttribute("subsampling_width", strides[1]);

    new_node.AddAttribute("dilation_height", dilations[0]);
    new_node.AddAttribute("dilation_width", dilations[1]);

    if(!is_depthwise) new_node.AddAttribute("groups", group);

    //TODO: what is NOTSET?
    new_node.AddAttribute("padding_mode", static_cast<int64_t>(1));
    ValueMoveInfo value_move_info2(InOutDefSlot{ArgType::kOutput, 0}, ArgType::kOutput, false, false);
    ORT_ENFORCE(MoveInputOutput(graph, *conv_node, new_node, value_move_info2, false).IsOK());
    ORT_ENFORCE(graph.RemoveNode(ni));
    //bool fused = false;

    if (optimizer_utils::CheckOutputEdges(graph, new_node, 1)) {
      const auto& next_node = *(new_node.OutputNodesBegin());
      float output_min;
      float output_max;
      bool has_clip = optimizer_utils::GetClipConstantMinMax(graph, next_node, output_min, output_max);
      if (has_clip) {
        new_node.AddAttribute("output_min", output_min);
        new_node.AddAttribute("output_max", output_max);
        ValueMoveInfo value_move_info3(InOutDefSlot{ArgType::kOutput, 0}, InOutDefSlot{ArgType::kOutput, 0});
        //const_cast
        Node* clip_node = graph.GetNode(next_node.Index());
        ORT_ENFORCE(MoveInputOutput(graph, *clip_node, new_node, value_move_info3, false).IsOK());
        ORT_ENFORCE(graph.RemoveNode(next_node.Index()));
      }
    }
  }

  ORT_RETURN_NEG_ONE_IF_ERROR(m->MainGraph().Resolve());
  {
    auto api_graph = MakeApiGraph(m->MainGraph(), cpu_allocator, kCpuExecutionProvider);
    onnx_layout_transformation::Optimize(*api_graph, /*allow_extended_ops*/ true);
  }
  auto model_proto = m->ToProto();
  ::ONNX_NAMESPACE::ShapeInferenceOptions options{true, 1, true};
  ::ONNX_NAMESPACE::shape_inference::InferShapes(model_proto,
                                                 OpSchemaRegistry::Instance(),
                                                 options);

  int fd = -1;
  ORT_RETURN_NEG_ONE_IF_ERROR(Env::Default().FileOpenWr(output_model_path, fd));
  if (!model_proto.SerializeToFileDescriptor(fd)) {
    return -1;
  }
  return 0;
}
