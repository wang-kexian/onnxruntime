import torch
import unittest

from ..pytorch_export_helpers import infer_input_info

# example usage from <ort root>/tools/python
# python -m unittest util/test/test_pytorch_export_helpers.py
# NOTE: at least on Windows you must use that as the working directory for all the imports to be happy


class TestModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TestModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x, min=0, max=1):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        step1 = self.linear1(x).clamp(min=min, max=max)
        step2 = self.linear2(step1)
        return step2


class TestInferInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._model = TestModel(1000, 100, 10)
        cls._input = torch.randn(1, 1000)

    def test_positional(self):

        # Construct our model by instantiating the class defined above
        # pred = model(x, 0, 1)
        input_names, inputs_as_tuple = infer_input_info(self._model, self._input, 0, 1)
        print(input_names)
        self.assertEqual(input_names, ['x', 'min', 'max'])

    def test_keywords(self):
        N, D_in, H, D_out = 1, 1000, 100, 10
        x = torch.randn(N, D_in)

        # Construct our model by instantiating the class defined above
        model = TestModel(D_in, H, D_out)
        input_names, inputs_as_tuple = infer_input_info(self._model, self._input, max=1, min=0)
        # pred = model(x, max=1, min=0)
        self.assertEqual(input_names, ['x', 'min', 'max'])
        self.assertEqual(inputs_as_tuple, (self._input, 0, 1))
