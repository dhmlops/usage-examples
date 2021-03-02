from PIL import Image
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

# TODO: Make it easily configurable
DATA = "data"
TRITON_IP = "localhost"
TRITON_HTTPS = "8000"
TRITON_GRPC = "8001"


# pre-processing
img = Image.open('data/7.png').convert('L')
img = img.resize((28, 28))
imgArr = np.asarray(img)/255
imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
imgArr= imgArr.astype(triton_to_np_dtype('FP32'))

## Client-Server GRPC
print("Using GRPC ... ")
triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=0)
inputs = []
inputs.append(grpcclient.InferInput('flatten_1_input', imgArr.shape, 'FP32'))
inputs[0].set_data_from_numpy(imgArr)
outputs = []
outputs.append(grpcclient.InferRequestedOutput('dense_3',class_count=0))
responses = []
responses.append(triton_client.infer('mnist',inputs,
                    request_id=str(1),
                    model_version='1',
                    outputs=outputs))

# post-proocessing
print (np.argmax(responses[0].as_numpy('dense_3')[0]))

## Client-Server HTTPS
print("Using HTTPS ... ")
triton_client = httpclient.InferenceServerClient(url='localhost:8000', verbose=0)
inputs = []
inputs.append(httpclient.InferInput('flatten_1_input', imgArr.shape, 'FP32'))
inputs[0].set_data_from_numpy(imgArr)
outputs = []
outputs.append(httpclient.InferRequestedOutput('dense_3',class_count=0))
responses = []
responses.append(triton_client.infer('mnist',inputs,
                    request_id=str(1),
                    model_version='1',
                    outputs=outputs))

# post-proocessing
print (np.argmax(responses[0].as_numpy('dense_3')[0]))
# TODO: Add in return of human-readable label
