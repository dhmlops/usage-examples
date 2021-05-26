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
import cv2


# server and model info
TRITON_IP = '0.0.0.0'
TRITON_HTTP_PORT = '8000'
MODEL_NAME = 'preprocess'
MODEL_VER = '1'

# hardcoded images for testing
IMAGES = ['data/7.png', 'data/7.png']



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
        resp = resp.json()
        model_name = resp['name']
        model_ver = resp['versions'][0]
        ip = resp['inputs'][0]
        op = resp['outputs'][0]
        input_name = ip['name']
        input_datatype = ip['datatype']
        input_shape = ip['shape']
        output_name = op['name']
        output_datatype = op['datatype']
        output_shape = op['shape']

        print("model_name: {}, model_ver: {}, input:{}, {}, {}, output: {}, {}, {}"
            .format(model_name, model_ver, input_name, input_datatype, input_shape, output_name, output_datatype, output_shape))

    return model_name, model_ver, input_name, input_datatype, input_shape, output_name, output_datatype, output_shape


def infer(img_paths, batch_size, model_name, model_ver, input_name, input_datatype, input_shape, output_name, output_datatype, output_shape):
    # /v2/models/{MODEL_NAME}/versions/{VERSION}/infer
    url = 'http://' + TRITON_IP + ':' + TRITON_HTTP_PORT + '/v2/models/' + model_name + '/versions/' + model_ver + '/infer'
    
    # read img and get tensor
    input_data = []
    for i in range(batch_size):
        img_path = img_paths[i]
        img = cv2.imread(img_path)
        input_data.append(img)

    input_data = np.array(input_data)
    print("input_data: ", input_data.shape)

    # random array
    # arrays = [np.random.randn(*MODEL_INPUT_SHAPE).astype(np.float32) for _ in range(batch_size)]
    # input_data = np.stack(arrays, axis=0)
    # print("input_data: ", input_data.shape)

    # construct inference request body
    infer_req = {
        "id": "optional-use-if-need-to-check-against-http-return",
        "inputs": [
            {
                "name": input_name,
                "shape": input_data.shape,
                "datatype": input_datatype,
                "data": input_data.tolist()
            }
        ],
        "outputs": [
            {
                "name": output_name,
            }
        ]
    }

    headers = {'Content-type': 'application/json'}
    resp = requests.post(url, data=json.dumps(infer_req), headers=headers)
    if resp.status_code == 200:
        print(".... inference SUCCESS ....")
        postprocessing(resp, batch_size)
    else:
        print('.... inference FAILED ....')
        handle_error(resp)


def handle_error(response):
    print("log your error / handle it...")
    print(response)


def postprocessing(response, batch_size):
    r = response.json()
    # r = json.loads(response.content)
    print("r: ", r)
    output_shape = r['outputs'][0]['shape']
    embeds = r['outputs'][0]['data']
    embeds = np.array(embeds)
    print("embeds: ", embeds.shape)

    embeds = np.reshape(embeds, output_shape)
    print("embeds: ", embeds.shape)


if __name__ == "__main__":
    get_server_health()
    model_name, model_ver, input_name, input_datatype, input_shape, output_name, output_datatype, output_shape = get_model_metadata()
    infer(IMAGES, len(IMAGES), model_name, model_ver, 
        input_name, input_datatype, input_shape, 
        output_name, output_datatype, output_shape)
