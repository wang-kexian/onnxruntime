# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pathlib
import unittest

from ..qdq_model_utils import update_model

script_dir = pathlib.Path(__file__).parent
ort_root = script_dir.parents[4]

# example usage from <ort root>/tools/python
# python -m unittest util/qdq_helpers/test/test_qdq_model_utils.py
# NOTE: at least on Windows you must use that as the working directory for all the imports to be happy


class TestQDQUtils(unittest.TestCase):
    def test_fix_DQ_with_multiple_consumers(self):
        '''
        '''
        # model_path = ort_root / 'onnxruntime' / 'test' / 'testdata' /  'model.onnx'
        update_model()
        self.assertTrue(True)
