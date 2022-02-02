import argparse
import logging
import pathlib

from .mobile_helpers import check_model_can_use_ort_mobile_pkg, usability_checker


def check_usability():
    parser = argparse.ArgumentParser(
        description=
        'Analyze an ONNX model to determine how well it will work in mobile scenarios, and whether '
        'it is likely to be able to use the pre-build ONNX Runtime Mobile Android or iOS package.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_path',
                        help='Path to required operators and types configuration used to build '
                             'the pre-built ORT mobile package.',
                        required=False,
                        type=pathlib.Path,
                        default=check_model_can_use_ort_mobile_pkg.get_default_config_path())
    parser.add_argument('--log_level', choices=['debug', 'info', 'warning', 'error'],
                        default='info', help='Logging level')
    parser.add_argument('model_path', help='Path to ONNX model to check', type=pathlib.Path)

    args = parser.parse_args()
    logger = logging.getLogger('default')
    if args.log_level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif args.log_level == 'info':
        logger.setLevel(logging.INFO)
    elif args.log_level == 'warning':
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.ERROR)

    usability_checker.analyze_model(args.model_path, skip_optimize=False, logger=logger)
    check_model_can_use_ort_mobile_pkg.run_check(args.model_path, args.config_path, logger)


if __name__ == '__main__':
    check_usability()
