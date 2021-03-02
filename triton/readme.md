# How to serve model and do model inference on Triton

## Overview
- Triton enables us to deploy our model in a way which any improvements to Triton can benefit a wider group of user. 
- These benefits include features of Triton like scalability, as well as minmising qualification effort when we adopt similar approach to model deployment and inference.
- This readme describes the basic way to deploy the model and inference. More advanced features will be added on subsequently.
- These are the general steps that you will need to use Triton. 

TODO: <add overview diagram>

1. Train the model, create config file and deploy to Triton Server.
2. Create own client and infer from deployed model in Triton Server.

We will be using a tensorflow example to walk through. The example codes are available in this github repo. 

## Train model
- Triton does not affect how the model is being trained.
- A Jupyter notebook example <add link> that trains a tensorflow model using MNIST data is made available to help you walkthu.
- The model is saved as "savedmodel" format.
- Triton also supports other ML framework and model format. 


## References
1. https://github.com/triton-inference-server/server
2. https://developer.nvidia.com/nvidia-triton-inference-server

