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
import torch


# data info
DATA = 'data/7.png'
INPUT_SHAPE = (28, 28)

# triton server info
TRITON_IP = '0.0.0.0'
TRITON_HTTP_PORT = '8000'


def get_server_health():
    # /v2/health/ready
    url = 'http://' + TRITON_IP + ':' + TRITON_HTTP_PORT + '/v2/health/ready'
    resp = requests.get(url)
    if resp.status_code == 200:
        print(".... server is READY ....")
    else:
        print(".... server NOT READY ....")


def get_model_metadata(model_name, model_ver):
    # /v2/models/{MODEL_NAME}/versions/{VERSION}
    url = 'http://' + TRITON_IP + ':' + TRITON_HTTP_PORT + '/v2/models/' + model_name + '/versions/' + model_ver
    resp = requests.get(url)
    if resp.status_code == 200:
        print(".... model metadata is fetched SUCCESS ....")
        for k, v in resp.json().items():
            print('{}: {}'.format(k, v))


def infer(data, model_name, model_ver, image_path, model_input_name, model_input_shape, model_input_datatype, model_output_name):
    # /v2/models/{MODEL_NAME}/versions/{VERSION}/infer
    url = 'http://' + TRITON_IP + ':' + TRITON_HTTP_PORT + '/v2/models/' + model_name + '/versions/' + model_ver + '/infer'



    # construct inference request body
    infer_req = {
        "id": "optional-use-if-need-to-check-against-http-return",
        "inputs": [
            {
                "name": model_input_name,
                "shape": model_input_shape,
                "datatype": model_input_datatype,
                "data": imgArr.tolist()
            }
        ],
        "outputs": [
            {
                "name": model_output_name,
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
        handle_error(resp)


def handle_error(resp):
    print("log your error / handle it...")
    print(resp)


def postprocessing(response):
    r = json.loads(response.content)
    probabilities = r['outputs'][0]['data']
    print('image is: {}'.format(np.argmax(probabilities)))


def preprocessing_tensorflow(image_path):
    '''
    Return (1, 28, 28, 1) with FP32
    '''
    img = Image.open(image_path).convert('L')
    img = img.resize(INPUT_SHAPE)
    imgArr = np.asarray(img) / 255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    imgArr = imgArr.astype(np.float32)
    return imgArr


def preprocessing_pytorch(image_path):
    '''
    Return (1, 1, 28, 28) with FP32
    '''
    img = Image.open(image_path).convert('L')
    img = img.resize(INPUT_SHAPE)
    imgArr = np.asarray(img) / 255

    # make it (1,1,28,28)
    imgArr = np.expand_dims(imgArr, 0)
    imgArr = np.expand_dims(imgArr, 0)
    imgArr = imgArr.astype(np.float32)
    # convert to tensor
    imgArr = torch.from_numpy(imgArr)
    
    return imgArr


if __name__ == "__main__":        
    print("======== tf ========")
    # tf model model info
    model_name = 'mnist_tf_savedmodel'
    model_ver = '2'
    model_input_name = 'flatten_1_input'
    model_input_shape = [1, 28, 28, 1]
    model_input_datatype = 'FP32'
    model_output_name = 'dense_3'

    # pre-process image to get numpy array of shape (1, 28, 28, 1)
    imgArr = preprocessing_tensorflow(DATA)

    get_server_health()    
    get_model_metadata(model_name, model_ver)
    infer(imgArr, model_name, model_ver, DATA, model_input_name, model_input_shape, model_input_datatype, model_output_name)

    print("======== pytorch ========")
    # pytorch info; note: pytorch does not have concept of input and output names. Just follow which are in config.pbtxt which follows a naming convention. 
    model_name = 'mnist_pytorch_pt'
    model_ver = '1'
    model_input_name = 'input__0'
    model_input_shape = [1, 1, 28, 28]
    model_input_datatype = 'FP32'
    model_output_name = 'output__0'

    # pre-process image to get numpy array of shape (1, 1, 28, 28); note the difference in the input format in config.pbtxt.
    imgArr = preprocessing_pytorch(DATA)

    get_server_health()
    get_model_metadata(model_name, model_ver)
    infer(imgArr, model_name, model_ver, DATA, model_input_name, model_input_shape, model_input_datatype, model_output_name)

    

    


