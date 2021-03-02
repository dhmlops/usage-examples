# How to do Experiment Logging on ClearML

## Overview
ClearML allows us to use it in different ways, as limiting our flexibility as little as possible. This article describes the most common way we would be using it.
1. Train on local laptop, log experiments on server
2. Train and log experiment on server
3. Hyperparameter Tuning: Train and log experiment on server


## Train on local laptop, log experiments on server
In this method, the developer codes and runs the code on his laptop as per normal. If the code is a training code, then the training will happen on the laptop. When the code is running, it will send updates to the ClearML Server and the server will log the progress in near realtime. A few things needs to be done for this to happen. Note, if you are submitting your codes to Kubernetes to train, its fine too but you have to consider the instructions below when building your docker image. Otherwise, consider the next [section](#Train and log experiment on server).

#### Install and configure clearml on laptop
Run the instaler
```bash
pip install clearml
```
copy [clearml.conf](clearml.conf) to ~/clearml.conf<br>
Change apt_server, web_server, files_server to the right IP/Name/Ports.<br> 
Credentials and secret_key can be retrieved by accessing the ClearML UI and then 'profile'.

```
api {
    # Notice: 'host' is the api server (default port 8008), not the web server.
    api_server: http://192.168.50.31:8008
    web_server: http://192.168.50.33:80
    files_server: http://192.168.50.32:8081
    # Credentials are generated using the webapp, http://192.168.50.88:8080/profile
    # Override with os environment: CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY
    credentials {"access_key": "0LKT6VX2TFXQF8EAXGX5", "secret_key": "NXDw)b6Y2^pjmh3o2qQAtL0cy3KL(O+%YdGWPk@Vhsw_hDcb7!"}
}
```
Getting your credentials: Screenshot of the screen at profile page<br>
<img src="https://github.com/dhmlops/usage-examples/raw/main/clearml/clearml_credentials.png" width="480">

#### Inject clearml code into your codes
First of all, to fulfil our goal of reproducibility, your codes should be checked in to Gitlab. ClearML is able to pick it up and reference for you.<br>
If you are following one of the following frameworks, and you already have reporting codes (E.g. tensorboard), you may use a two liner to perform a 'automagikal logging'. (Note: You can still choose to manual log if there's issues or there's non standard stuff to log)
-  PyTorch(incl' ignite/lightning), Tensorflow, Keras, AutoKeras, XGBoost and Scikit-Learn
```python
#Put this at the beginning of your codeset
from clearml import Task
task = Task.init(project_name='My Project Name - Event Extraction', task_name='My Task Name - Dygie')
```
```python
# A typical pytorch with tensorboard reporting would have something as follows, these will be captured by ClearML automatically
from torch.utils.tensorboard import SummaryWriter
.
.
writer = SummaryWriter('runs')
.
.
writer.add_scalar('Train/Loss', loss.data.item(), niter)
.
.
```
A set of working codes is already made available at [pytorch_mnist.py](pytorch_mnist.py).


If you are not following one of the above frameworks, and your code is in Python, you can use the manual approach.
Details of the manual approach can be found here. <br>
Tutorial:https://allegro.ai/clearml/docs/docs/tutorials/tutorial_explicit_reporting.html <br>
Examples: https://allegro.ai/clearml/docs/rst/examples/explicit_reporting/index.html

## Train and log experiment on server
The most typical way for us to train on the server is to either just SSH into the server and run our codes, or convert our codes into Docker and then submit the whole image as a Kubernetes job. The former is obviously not the way out simply because there is no telling who will use which GPU. The latter sounds better but still require some work. This section shows you how to simply from your code, add one line, run the code and sit back and watch the rest happening.

To make this happen, the idea is to tell ClearML that you are creating a experiment run and then ClearML will terminate the code from running locally but bring it up to the server. For ClearML to do this, it needs to be able to pull the entire repo of codes. So...this means the codes must pushed inside a git repo before this works, and that's pretty much everything...i hope.

#### Install and configure clearml on laptop
Run the instaler
```bash
pip install clearml
```
copy [clearml.conf](clearml.conf) to ~/clearml.conf<br>
Change apt_server, web_server, files_server to the right IP/Name/Ports.<br> 
Credentials and secret_key can be retrieved by accessing the ClearML UI and then 'profile'.

```
api {
    # Notice: 'host' is the api server (default port 8008), not the web server.
    api_server: http://192.168.50.31:8008
    web_server: http://192.168.50.33:80
    files_server: http://192.168.50.32:8081
    # Credentials are generated using the webapp, http://192.168.50.88:8080/profile
    # Override with os environment: CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY
    credentials {"access_key": "0LKT6VX2TFXQF8EAXGX5", "secret_key": "NXDw)b6Y2^pjmh3o2qQAtL0cy3KL(O+%YdGWPk@Vhsw_hDcb7!"}
}
```
Getting your credentials: Screenshot of the screen at profile page<br>

#### Inject clearml code into your codes
```python
#Put this at the beginning of your codeset
from clearml import Task
task = Task.init(project_name='My Project Name - Event Extraction', task_name='My Task Name - Dygie')

task.set_base_docker("nvidia/cuda:10.1-runtime-ubuntu18.04 -e GIT_SSL_NO_VERIFY=true")
task.execute_remotely(queue_name="gpu", exit_process=True)
```
Now the first two lines are the standard lines that you need to put for ClearML<br>
The third line allows you to indicate which docker base to use, and the environment variables you want to set<br>
The last line simply tell ClearML to terminate the local run and assign this task to the GPU queue. Quite the ingenuity if you ask me<br>
You should see an output similar to following
```bash
(venv) jax@Kahs-MacBook-Pro pytorchmnist % python3 pytorch_mnist_task.py
ClearML Task: created new task id=43f65db3b3e54801b33a3eaa2546427a
ClearML results page: http://192.168.50.33:80/projects/2198e4eb6f664fb29e35e2bb249796ed/experiments/43f65db3b3e54801b33a3eaa2546427a/output/log
2021-03-02 00:31:14,777 - clearml - WARNING - Switching to remote execution, output log page http://192.168.50.33:80/projects/2198e4eb6f664fb29e35e2bb249796ed/experiments/43f65db3b3e54801b33a3eaa2546427a/output/log
2021-03-02 00:31:14,777 - clearml - WARNING - Terminating local execution process
```
The sample code used here is available at [pytorch_mnist_task.py](pytorch_mnist_task.py)<br>
## Hyperparameter Tuning: Train and log experiment on server
In this scenario, we already developed a training code and we are just running some hyperparameter search. The most typical way is a Grid Search, more complex methods are Bayesian Optimisation. Typically, someone would write a gridsearch and then start running it by submitting it as a job to Kubernetes. This section tries to cover as many scenarios as possible.

#### Standard Gridsearch
The standard gridsearch is where you simply perform what is described in [Train and log experiment on server](#train-and-log-experiment-on-server), but in this case, you need some manual explicit reporting to report the results of the gridsearch. Quite frankly, given ClearML, i would like you to try the following subsection instead....so yeah, i am underplaying this section and emphasizing on the next.

#### Hyperparameter using ClearML
1. Choose a search strategry
2. 

ref: https://allegro.ai/clearml/docs/docs/examples/optimization/hyper-parameter-optimization/examples_hyperparam_opt.html





