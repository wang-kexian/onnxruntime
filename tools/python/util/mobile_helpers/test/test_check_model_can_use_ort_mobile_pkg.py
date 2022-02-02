import logging
import onnx
import pathlib
import unittest

from onnx import shape_inference
from testfixtures import LogCapture

from ..check_model_can_use_ort_mobile_pkg import check_model_supported_by_mobile_package

# example usage from <ort root>/tools/python
# python -m unittest util/mobile_helpers/test/test_usability_checker.py
# NOTE: at least on Windows you must use that as the working directory for all the imports to be happy

script_dir = pathlib.Path(__file__).parent
ort_root = script_dir.parents[4]

ort_package_build_config_filename = \
    ort_root / 'tools' / 'ci_build' / 'github' / 'android' / 'mobile_package.required_operators.config'


def _create_logger():
    logger = logging.getLogger('default')
    logger.setLevel(logging.DEBUG)
    return logger


class TestMobilePackageModelChecker(unittest.TestCase):
    @classmethod
    def open_model(cls, model_path: pathlib.Path):
        model = onnx.load(str(model_path))
        return shape_inference.infer_shapes(model)

    def test_supported_model(self):
        with LogCapture() as log_capture:
            logger = _create_logger()
            model_path = ort_root / 'onnxruntime' / 'test' / 'testdata' / 'ort_github_issue_4031.onnx'
            model = self.open_model(model_path)
            supported = check_model_supported_by_mobile_package(model, ort_package_build_config_filename, logger)
            self.assertTrue(supported)

            # print(log_capture)
            log_capture.check_present(
                ('default', 'INFO', 'Model is most likely supported. Note that this check is not comprehensive '
                 'so testing to validate is still required.'),
            )

    def test_model_invalid_opset(self):
        with LogCapture() as log_capture:
            logger = _create_logger()
            model_path = ort_root / 'onnxruntime' / 'test' / 'testdata' / 'mnist.onnx'
            model = self.open_model(model_path)
            supported = check_model_supported_by_mobile_package(model, ort_package_build_config_filename, logger)
            self.assertFalse(supported)

            # print(log_capture)
            log_capture.check_present(
                ('default', 'INFO', 'Model uses ONNX opset 8.'),
                ('default', 'INFO', 'The pre-built package only supports ONNX opsets [12, 13, 14, 15].')
            )

    def test_model_unsupported_op_and_types(self):
        with LogCapture() as log_capture:
            logger = _create_logger()
            model_path = ort_root / 'onnxruntime' / 'test' / 'testdata' / 'sequence_insert.onnx'

            # Model uses opset 11 which is not supported in the mobile package. Update to supported opset first
            # Note: Ideally this would use update_onnx_opset however the ONNX opset update tools isn't working with
            # that at the moment (fix on ONNX side is pending).
            # For the sake of this test do a manual update. As the spec hasn't changed for the operators in the model
            # this is safe.
            # from ...onnx_model_utils import update_onnx_opset
            # model = update_onnx_opset(model_path, 13, logger=logger)
            # model = shape_inference.infer_shapes(model, strict_mode=True)
            model = self.open_model(model_path)
            model.opset_import[0].version = 13

            supported = check_model_supported_by_mobile_package(model, ort_package_build_config_filename, logger)
            self.assertFalse(supported)

            # print(log_capture)
            log_capture.check_present(
                ('default', 'DEBUG', 'Data type sequence_type of graph input input_seq is not supported.'),
                ('default', 'INFO', 'Unsupported operators:'),
                ('default', 'INFO', '  ai.onnx:13:SequenceInsert'),
            )
