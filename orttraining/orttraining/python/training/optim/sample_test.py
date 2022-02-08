
import torch
import adam
import onnx
import onnxruntime
import numpy as np
import copy
from onnxruntime.capi import _pybind_state as C

# neural network def
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, model_input):
        out = self.fc1(model_input)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_initial_inputs_and_outputs(learning_rate, step, model):
    ort_inputs = {}
    ort_output_names = []
    ort_output_names_to_param_map = {}
    for name, param in model.named_parameters():
        ort_inputs[name+'.learning_rate'] = np.array([learning_rate], dtype=np.float32)
        if name+'.step' not in ort_inputs:
            ort_inputs[name+'.step'] = np.array([step], dtype=np.int64)
        
        ort_inputs[name] = to_numpy(param.data)
        ort_inputs[name+'.gradient'] = to_numpy(param.grad)
        ort_inputs[name+'.exp_avg'] = np.zeros(list(param.shape), dtype=np.float32)
        ort_inputs[name+'.exp_avg_sq'] = np.zeros(list(param.shape), dtype=np.float32)
        ort_inputs[name+'.mixed_precision'] = np.array([], dtype=np.float16)
        ort_inputs[name+'.loss_scaler'] = np.array([], dtype=np.float32)
        ort_inputs[name+'.global_gradient_norm'] = np.array([], dtype=np.float32)
        ort_inputs[name+'.should_update'] = np.array([True], dtype=np.bool)

        ort_output_names.extend([name+'.step.out',
                                name+'.exp_avg.out',
                                name+'.exp_avg_sq.out',
                                name+'.out',
                                name+'.gradient.out',
                                name+'.mixed_precision.out'])

        ort_output_names_to_param_map[name+'.out'] = param

    return ort_inputs, ort_output_names, ort_output_names_to_param_map

def update_inputs_and_weights(ort_inputs, ort_output_names, ort_output_names_to_param_map, ort_outs):
    # weight updates
    for name, value in zip(ort_output_names, ort_outs):
        if name in ort_output_names_to_param_map:
            with torch.no_grad():
                ort_output_names_to_param_map[name].copy_(torch.from_numpy(value))

        # update the inputs for the next run
        # inputs are named same as outputs minus the last four chars ('.out')
        ort_inputs[name[:-4]] = value

# initialize
batch_size = 16
input_size = 10
num_classes = 10
hidden_size = 20
model = NeuralNet(input_size, hidden_size, num_classes)
model_copy = copy.deepcopy(model) # make model copy

# define optimizer and export it as a graph
optimizer = adam.Adam(model.named_parameters())
onnx_model = optimizer.export()
onnx.save(onnx_model, 'optimizer_model.onnx')

# weight update step
ort_inputs = {}
ort_output_names = []
ort_output_names_to_param_map = {}

learning_rate = 0.001
step = 1

batches = [torch.randn(batch_size, input_size) for _ in range(5)]

available_providers = C.get_available_providers()
ort_session = onnxruntime.InferenceSession('optimizer_model.onnx', providers=available_providers)

# training loop
for batch in batches:
    # run forward+backward
    output = model(batch)
    loss = output.sum()
    loss.backward()

    # prepare inputs and outputs
    if not ort_inputs:
        ort_inputs, ort_output_names, ort_output_names_to_param_map = get_initial_inputs_and_outputs(
            learning_rate, step, model
        )

    # run optimizer graph
    ort_outs = ort_session.run(ort_output_names, ort_inputs)

    # weight update
    update_inputs_and_weights(ort_inputs, ort_output_names, ort_output_names_to_param_map, ort_outs)

    # reset gradients
    model.zero_grad()
