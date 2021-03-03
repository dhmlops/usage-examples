#!/bin/bash

echo starting triton with cpu mode

docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v /media/ky/Data/workspace_aieng/triton-server/model_repo:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models
