'''
RESTFUL APIs: https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md
This python file includes examples to do RESTFUL calls to Triton Server for
1. server health
2. model metadata
3. inference

You can also use Triton Client Libraries to do REST/GRPC calls to Triton Server. Library available in C++ / Python.
More REST APIs can be found in the link above.
'''


import requests
import json
from PIL import Image
import numpy as np


# data info
DATA = 'data/7.png'
INPUT_SHAPE = (28, 28)

# server and model info
TRITON_IP = '0.0.0.0'
TRITON_HTTP_PORT = '13010'
MODEL_NAME = 'mnist_tf_savedmodel'
MODEL_VER = '1'
MODEL_INPUT_NAME = 'flatten_1_input'
MODEL_INPUT_SHAPE = [1, 28, 28, 1]
MODEL_INPUT_DATATYPE = 'FP32'
MODEL_OUTPUT_NAME = 'dense_3'


def get_server_health():
    # /v2/health/ready
    url = 'http://' + TRITON_IP + ':' + TRITON_HTTP_PORT + '/v2/health/ready'
    resp = requests.get(url)
    if resp.status_code == 200:
        print(".... server is READY ....")
    else:
        print(".... server NOT READY ....")


def get_model_metadata():
    # /v2/models/{MODEL_NAME}/versions/{VERSION}
    url = 'http://' + TRITON_IP + ':' + TRITON_HTTP_PORT + '/v2/models/' + MODEL_NAME + '/versions/' + MODEL_VER
    resp = requests.get(url)
    if resp.status_code == 200:
        print(".... model metadata is fetched SUCCESS ....")
        for k, v in resp.json().items():
            print('{}: {}'.format(k, v))


def infer(image_path):
    # /v2/models/{MODEL_NAME}/versions/{VERSION}/infer
    url = 'http://' + TRITON_IP + ':' + TRITON_HTTP_PORT + '/v2/models/' + MODEL_NAME + '/versions/' + MODEL_VER + '/infer'

    # pre-process image to get numpy array of shape (1, 28, 28, 1)
    imgArr = preprocessing(image_path)

    # construct inference request body
    infer_req = {
        "id": "optional-use-if-need-to-check-against-http-return",
        "inputs": [
            {
                "name": MODEL_INPUT_NAME,
                "shape": MODEL_INPUT_SHAPE,
                "datatype": MODEL_INPUT_DATATYPE,
                "data": imgArr.tolist()
            }
        ],
        "outputs": [
            {
                "name": MODEL_OUTPUT_NAME,
            }
        ]
    }

    headers = {'Content-type': 'application/json'}
    resp = requests.post(url, data=json.dumps(infer_req), headers=headers)
    if resp.status_code == 200:
        print(".... inference SUCCESS ....")
        postprocessing(resp)
    else:
        print('.... inference FAILED ....')
        handle_error()


def handle_error():
    print("log your error / handle it...")


def postprocessing(response):
    r = json.loads(response.content)
    probabilities = r['outputs'][0]['data']
    print('image is: {}'.format(np.argmax(probabilities)))


def preprocessing(image_path):
    '''
    Return (1, 28, 28, 1) with FP32
    '''
    img = Image.open(image_path).convert('L')
    img = img.resize(INPUT_SHAPE)
    imgArr = np.asarray(img) / 255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    imgArr = imgArr.astype(np.float32)
    return imgArr


if __name__ == "__main__":
    get_server_health()
    get_model_metadata()
    infer(DATA)

