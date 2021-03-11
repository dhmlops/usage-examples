# How to serve model and do model inference on Triton

## Overview
Triton enables us to deploy and infer from the model in an effective and scalable manner. This readme describes the basic way to deploy the model and inference. More advanced features will be added on subsequently. 

These are the general steps that you will need to use Triton. 
1. Install dependencies.
2. Train the model, create config file and deploy to Triton Server.
3. Create own client and infer from deployed model in Triton Server.

We will be using a tensorflow example to walk through. The example codes are available in this github repo. 

![alt text](https://github.com/dhmlops/usage-examples/blob/main/triton/images/overview)


### Install dependencies [Optional]
Install these if you are using Triton Client library to do REST/GRPC calls to Triton Server.
Alternatively, you can construct your own REST call e.g. using Python requests to do RESTFUL call to Triton Server.

```bash
# install triton client libraries
pip3 install nvidia-pyindex
pip3 install tritonclient[all]

# install client dependencies
pip3 install Pillow
```


### Train model
- Triton does not affect how the model is being trained.
- A Jupyter notebook example https://github.com/dhmlops/usage-examples/blob/main/triton/model%20training/MNIST_TF_1.ipynb that trains a tensorflow model using MNIST data is made available to help you walkthu.
- The model is saved as "savedmodel" format.
- Triton also supports other ML framework and model format. 
- TODO: add the table.

### Create config.pbtxt
This is a sample of the config.pbtxt based on the tensorflow example.

```
name: "mnist_tf_savedmodel"
platform: "tensorflow_savedmodel"
max_batch_size: 32
input [
    {
        name: "flatten_1_input"
        data_type: TYPE_FP32
        format: FORMAT_NHWC
        dims: [28, 28, 1]
    }
]
output [
    {
        name: "dense_3"
        data_type: TYPE_FP32
        dims: [10]
    }
]
```
- “name” must be the same as its parent folder
- “max_batch_size”: TBC
- “input”: The MNIST model takes in an image of (28, 28, 1) -> H x W x Channel.
- “output: The MNIST model output probability of 10 classes.
TODO: add in the table for different platform values.

You can get the model info through these methods (Refer to this notebook https://github.com/dhmlops/usage-examples/blob/main/triton/model%20training/MNIST_TF_1.ipynb for working codes):
```python
# for tensorflow model
model.input
<KerasTensor: shape=(None, 28, 28, 1) dtype=float32 (created by layer 'flatten_1_input'
.....

model.output

.....
<KerasTensor: shape=(None, 10) dtype=float32 (created by layer 'dense_3')>
```

### Deploy to Triton Server
Deploy the trained model and its config file based on the folder structure shown in the diagram below.
An example of the trained model and its config file is available at https://github.com/dhmlops/usage-examples/tree/main/triton/sample_model_repo/mnist_tf_savedmodel.

![alt text](https://github.com/dhmlops/usage-examples/blob/main/triton/images/model_repor_folder_structure.png)

Start the Triton Server after adding the model using the bash script provided. 
https://github.com/dhmlops/usage-examples/blob/main/triton/startserver_cpu.sh

You should see the model deployed with status "READY".

```
I0302 03:06:02.705883 1 server.cc:533] 
+---------------------+---------+--------+
| Model               | Version | Status |
+---------------------+---------+--------+
| mnist_tf_savedmodel | 1       | READY  |
+---------------------+---------+--------+
```


### Create Client
After the model is deployed successfully, we can create client to do model inference. 
- Client code in general contains:
  - Pre-processing of data to model input shape and data type.
  - Call Triton Server (with REST / GRPC). 
  - Post-processing of Triton Server’s response
  - 
Here's two options for the client REST/GRPC:
1. Using Triton Client Library
2. Construct your own REST e.g. using Python requests

#### Using Triton Client Library
``` python
# pre-processing of data to get required shape and data type
... codes...

# Sample codes for Client-Server HTTPS; similarly for GRPC as in the client.py sample
triton_client = httpclient.InferenceServerClient(url=URL_HTTP, verbose=0)
inputs = []
inputs.append(httpclient.InferInput('flatten_1_input', imgArr.shape, 'FP32'))
inputs[0].set_data_from_numpy(imgArr)
outputs = []
outputs.append(httpclient.InferRequestedOutput('dense_3', class_count=0))
responses = []
responses.append(triton_client.infer(MODEL,inputs,
                    request_id=str(1),
                    model_version=MODEL_VER,
                    outputs=outputs))

# post-processing, can be anything like getting the human-readable class
... codes...

```

Run the client.py https://github.com/dhmlops/usage-examples/tree/main/triton/client. Note that the path to data is hardcoded, so you can simply run the python file with ```python python3 client.py```
You should see this output.

``` bash
# run client.py to do inference with 7.png data.
python3 client.py
Using GRPC ... 
7
Using HTTPS ... 
7
```
#### Construct own REST
The REST APIs are available here. https://github.com/kubeflow/kfserving/blob/master/docs/predict-api/v2/required_api.md
Refer to this sample codes. https://github.com/dhmlops/usage-examples/tree/main/triton/client_REST

## References
1. https://github.com/triton-inference-server/server
2. https://developer.nvidia.com/nvidia-triton-inference-server

