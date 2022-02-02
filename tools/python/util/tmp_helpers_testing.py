import argparse
import logging
import onnx.shape_inference
import os
import pathlib
import torch
import torchvision

# setup logging
FUNC_NAME_WIDTH = 24
FORMAT = '%(funcName)' + str(FUNC_NAME_WIDTH) + 's %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('default')
logger.setLevel(logging.INFO)


def get_model():
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    # model.eval()
    qmodel = torchvision.models.quantization.mobilenet_v3_large(pretrained=True, progress=True, quantize=True)
    qmodel.eval()

    return qmodel


def load_and_preprocess_image(image_filename):
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(image_filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch


def run_with_pt(model, input_batch):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        # test execution
        output = model(input_batch)

        torch.onnx.export(model, input_batch, 'mobilenet_v3_large_quant.onnx',
                          input_names=['image'], output_names=['scores'],
                          opset_version=13,
                          do_constant_folding=False,
                          training=False,
                          export_params=True,
                          keep_initializers_as_inputs=False
                          )

    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    return probabilities


def download_file(url, filename):
    import urllib
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)


def download_data():
    # download_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    download_file("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")


def test_run():
    qmodel = get_model()
    image = r'D:\mlperf\imagenet2012\val\ILSVRC2012_val_00000001.JPEG'

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # load image, preprocess, create batch with single input
    input_batch = load_and_preprocess_image(image)

    input_names, inputs = infer_input_info(qmodel, input_batch)
    export_module(qmodel, inputs, input_names, ['scores'], 'mobilenet_v3_large.q.onnx')

    probabilities = run_with_pt(qmodel, image)

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


def run_helpers():
    download_data()
    test_run()

    logger.setLevel(logging.DEBUG)

    checker(r'D:\MobileBuildPackageModels\Converted\IndividualModels\resnet50_v1\resnet50_v1.onnx')
    # checker(r'C:\Users\scmckay\Downloads\mlperf_models_202103\mobile\mobilenet_edgetpu\mobilenet_edgetpu_224_1.0_float.onnx')
    # checker(r'C:\Users\scmckay\Downloads\mlperf_models_202103\mobile\mobilenet_edgetpu\mobilenet_edgetpu_224_1.0_float-int8.onnx')
    # checker(r'C:\Users\scmckay\Downloads\mlperf_models_202103\mobile\mobilenet_edgetpu\mobilenet_edgetpu_224_1.0-qdq.onnx')


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='''Analyze an ONNX model for usage with the ORT mobile'''
    )

    parser.add_argument('--log_level', choices=['debug', 'info', 'warning', 'error'],
                        default='info', help='Logging level')

    parser.add_argument('--optimize', action='store_true',
                        help='Optimize the model using ONNX Runtime before analyzing.')
    parser.add_argument('model_path', type=pathlib.Path, help='Provide path to ONNX model')

    return parser.parse_args()


def analyze_model():
    args = parse_args()

    model_path = args.model_path.resolve()
    if args.log_level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif args.log_level == 'info':
        logger.setLevel(logging.INFO)
    elif args.log_level == 'warning':
        logger.setLevel(logging.warning)
    else:
        logger.setLevel(logging.ERROR)

    if args.optimize:
        model_path = optimize_model(model_path)

    run_helpers()
    checker(str(model_path))


if __name__ == '__main__':
    analyze_model()
