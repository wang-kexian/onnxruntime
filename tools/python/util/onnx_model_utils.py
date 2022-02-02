import logging
import onnx
import onnxruntime as ort
import pathlib

from onnx import version_converter


def iterate_graph_per_node_func(graph, per_node_func, **func_args):
    '''
    Iterate the graph including subgraphs calling the per_node_func for each node.
    :param graph: Graph to iterate
    :param per_node_func: Function to call for each node. Signature is fn(node: onnx:NodeProto, **kwargs)
    :param func_args: The keyword args to pass through.
    '''

    for node in graph.node:
        per_node_func(node, **func_args)
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                iterate_graph_per_node_func(attr.g, per_node_func, **func_args)


def iterate_graph_per_graph_func(graph, per_graph_func, **func_args):
    '''
    Iterate the graph including subgraphs calling the per_graph_func for each Graph.
    :param graph: Graph to iterate
    :param per_graph_func: Function to call for each graph. Signature is fn(node: onnx:GraphProto, **kwargs)
    :param func_args: The keyword args to pass through.
    '''

    per_graph_func(graph, **func_args)

    for node in graph.node:
        # recurse into subgraph for control flow nodes (Scan/Loop/If)
        for attr in node.attribute:
            if attr.HasField('g'):
                iterate_graph_per_graph_func(attr.g, per_graph_func, **func_args)


def update_onnx_opset(model_path: pathlib.Path, opset: int, out_path: pathlib.Path = None,
                      logger: logging.Logger = None):
    """
    Helper to update the opset of a model using onnx version_converter. Target opset must be greater than current opset.
    Model is saved to the original location with the '.onnx' extension replaced with '.opset<opset>.onnx'.
    :param model_path: Path to model to update
    :param opset: Opset to update model to
    :param out_path: Optional output path for updated model.
    :param logger: Optional logger for diagnostic output
    :returns: Updated onnx.ModelProto
    """

    model_path_str = str(model_path.resolve(strict=True))
    if logger:
        logger.info("Updating %s to opset %d", model_path_str, opset)

    model = onnx.load(model_path_str)
    new_model = version_converter.convert_version(model, opset)

    # # save with .onnx -> .opsetX.onnx
    # if not out_path:
    #     out_path = str(model_path.with_suffix(f'.opset{opset}.onnx'))
    if out_path:
        onnx.save(new_model, str(out_path))
        if logger:
            logger.info("Saved updated model to %s", model_path)

    return new_model


def optimize_model(model_path: pathlib.Path,
                   output_path: pathlib.Path = None,
                   level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                   log_level: int = 3):
    '''
    Optimize an ONNX model using ONNX Runtime to the specified level
    :param model_path: Path to ONNX model
    :param output_path: Optional output path. If not specified the '.onnx' extension of model_path will be replaced
                        with '.<optimization level>.optimized.onnx'. e.g. '.basic.optimized.onnx'
    :param level: onnxruntime.GraphOptimizationLevel to use. Default is ORT_ENABLE_BASIC.
    :param log_level: Log level. Defaults to Error (3) so we don't get output about unused initializers being removed.
                      Warning (2) or Info (1) may be desirable in some scenarios.
    :return: output_path that was used.
    '''

    if not output_path:
        output_path = model_path.with_suffix(".{}.optimized.onnx".format(str(level)))

    so = ort.SessionOptions()
    so.optimized_model_filepath = str(output_path)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.log_severity_level = log_level

    # create session to optimize
    _ = ort.InferenceSession(str(model_path), so, providers=['CPUExecutionProvider'])

    return output_path


def _replace_symbolic_dim_value(graph: onnx.GraphProto, **kwargs):
    param_to_replace = kwargs['dim_param']
    value = kwargs['value']

    def update_dim_values(value_infos):
        for vi in value_infos:
            if vi.type.HasField("tensor_type"):
                shape = vi.type.tensor_type.shape
                if shape:
                    for dim in shape.dim:
                        if dim.HasField('dim_param') and dim.dim_param == param_to_replace:
                            dim.Clear()
                            dim.dim_value = value

    update_dim_values(graph.input)
    update_dim_values(graph.output)
    update_dim_values(graph.value_info)


def _make_dim_param_fixed(graph: onnx.GraphProto, param_name: str, value: int):
    '''
    Iterate all values in the graph, replacing dim_param in a tensor shape with the provided value.
    :param graph: GraphProto to update
    :param dim_param: dim_param to set
    :param value: value to use
    '''
    iterate_graph_per_graph_func(graph, _replace_symbolic_dim_value, dim_param=param_name, value=value)


def _make_input_shape_fixed(graph: onnx.GraphProto, input_name, fixed_shape: [int]):
    '''
    Update the named graph input to set shape to the provided value. This can be used to set unknown dims as well
    as to replace dim values.
    If setting the shape replaces a dim_param, apply that update to the rest of the graph and any subgraphs.
    :param graph: Graph to update
    :param input_name: Name of graph input to update.
    :param fixed_shape: Shape to use.
    '''

    for i in graph.input:
        if i.name == input_name:
            if not i.type.HasField("tensor_type"):
                raise ValueError(f'Input {input_name} is not a tensor')

            # graph input are required to have a shape to provide the rank
            shape = i.type.tensor_type.shape
            if len(shape.dim) != len(fixed_shape):
                raise ValueError(
                    f'Rank mismatch. Existing:{len(shape.dim)} Replacement:{len(fixed_shape)}')

            idx = 0
            for dim in shape.dim:
                # check any existing fixed dims match
                if dim.HasField('dim_value'):
                    if dim.dim_value != fixed_shape[idx]:
                        raise ValueError(
                            f"Can't replace existing fixed size of {dim.dim_value} with {fixed_shape[idx]} "
                            f"for dimension {idx + 1}")
                elif dim.HasField('dim_param'):
                    # replacing a dim_param so have to do that through the entire graph
                    _make_dim_param_fixed(graph, dim.dim_param, fixed_shape[idx])
                else:
                    # replacing an unknown dim
                    dim.Clear()
                    dim.dim_value = fixed_shape[idx]

                idx += 1
            return

    raise ValueError(f'Input {input_name} was not found in graph inputs. '
                     f'Valid input names are: {",".join([i.name for i in graph.input])}')


def make_dynamic_shape_fixed(model_path: pathlib.Path,
                             output_path: pathlib.Path,
                             dim_param: str = None, dim_value: int = -1,
                             input_name: str = None, input_shape: [int] = None):
    model = onnx.load(str(model_path))
    if dim_param:
        _make_dim_param_fixed(model.graph, dim_param, dim_value)
    else:
        _make_input_shape_fixed(model.graph, input_name, input_shape)

    onnx.save(model, str(output_path))


def _create_producer_consumer_link(node_to_producers: dict, node_to_consumers: dict,
                                   producer: onnx.NodeProto, consumer: onnx.NodeProto):
    '''
    Create links between two nodes for a value produced by one and consumed by the other.
    :param node_to_producers: Map of NodeProto to set of nodes that produce values the node consumes as inputs.
    :param node_to_consumers: Map of NodeProto to set of nodes that consume values the node produces as outputs.
    :param producer: Producer node
    :param consumer: Consumer node
    '''

    if consumer not in node_to_producers:
        node_to_producers[consumer] = set()

    if producer not in node_to_consumers:
        node_to_consumers[producer] = set()

    # add entry mapping this node to the producer of this input
    node_to_producers[consumer].add(producer)
    node_to_consumers[producer].add(consumer)


def _map_node_dependencies(graph: onnx.GraphProto, node_to_producers: dict, node_to_consumers: dict):
    graph_inputs = set([i.name for i in graph.input])
    initializers = set([i.name for i in graph.initializer])

    # map of value name to node that creates it. copy parent values but override if values get shadowed
    producers = {}

    implicit_inputs = set()

    def is_local_value(value):
        return value in producers or value in initializers or value in graph_inputs

    for node in graph.node:
        inputs = [i for i in node.input]

        for attr in node.attribute:
            if attr.HasField('g'):
                subgraph_implicit_inputs = _map_node_dependencies(attr.g, node_to_producers, node_to_consumers)
                inputs += subgraph_implicit_inputs

        for i in inputs:
            if not i:
                # missing optional input
                continue

            if is_local_value(i):
                if i in producers:
                    producer = producers[i]
                    _create_producer_consumer_link(node_to_producers, node_to_consumers, producer, node)
            else:
                # not produced above us, not in initializers for this graph. may be graph input or initializer
                # in parent graph
                implicit_inputs.add(i)

        for o in node.output:
            producers[o] = node

    return implicit_inputs


def get_producer_consumer_maps(graph: onnx.GraphProto):
    '''
    Get maps for connections between the nodes that produces each value and the nodes that consumer the value.
    Processing includes subgraphs. As the map key is a Node instance from the Graph there should be no ambiguity.
    :param graph: Graph to process.
    :return: Tuple with two maps.
             First is node_to_producers map of a node to set of all nodes producing input it consumes.
             Second is node_to_consumers map of a node to set of all nodes consuming output it creates.
             e.g. NodeA and NodeB provide inputs to NodeC. NodeC provides input to NodeD
             node_to_consumers[NodeA] = set([NodeC])
             node_to_consumers[NodeB] = set([NodeC])
             node_to_producers[NodeC] = set([NodeA, NodeB])
             node_to_consumers[NodeC] = set([NodeD])
             node_to_producers[NodeD] = set([NodeC])
    '''

    # use a hash of the object id for NodeProto.
    # we need this for the partitioning checker where we keep maps with nodes as the key.
    onnx.NodeProto.__hash__ = lambda self: id(self)

    node_to_producers = {}  # map of node instance to nodes producing input values it consumes
    node_to_consumers = {}  # map of node instance to nodes consuming output values it produces

    implicit_inputs = _map_node_dependencies(graph, node_to_producers, node_to_consumers)

    # top level graph should have no implicit inputs
    if implicit_inputs:
        raise ValueError('This appears to be an invalid model with missing inputs of '
                         f'{",".join(sorted(implicit_inputs))}')

    return node_to_producers, node_to_consumers


def is_fixed_size_tensor(value: onnx.ValueInfoProto):
    '''
    Check if value is a tensor with a fixed shape.
    :param value: onnx.ValueInfoProto to check
    :return: true if value is a tensor, with a shape, where all dimensions have fixed values.
    '''

    is_fixed = False
    if value.type.HasField("tensor_type"):
        shape = value.type.tensor_type.shape
        if shape:
            is_fixed = True  # scalar has no dims so set to True and unset if we hit a dim without a valid value
            for dim in shape.dim:
                if dim.HasField('dim_value') and dim.dim_value > 0:
                    continue

                # anything else means it's a dynamic value
                is_fixed = False
                break

    return is_fixed


def get_optimization_level(level):
    '''Convert string to GraphOptimizationLevel.'''
    if level == 'disable':
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if level == 'basic':
        # Constant folding and other optimizations that only use ONNX operators
        return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if level == 'extended':
        # Optimizations using custom operators, excluding NCHWc and NHWC layout optimizers
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if level == 'all':
        return ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    raise ValueError('Invalid optimization level of ' + level)
