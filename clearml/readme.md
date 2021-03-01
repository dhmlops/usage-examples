# Using ClearML Server

## Overview
ClearML allows us to use it in different ways, as limiting our flexibility as little as possible. This article describes the most common way we would be using it.
1. Train on local laptop, log experiments on server
2. Train on server, log experiments on server


## Train on local laptop, log experiments on server
In this method, the developer codes and runs the code on his laptop as per normal. If the code is a training code, then the training will happen on the laptop. When the code is running, it will send updates to the ClearML Server and the server will log the progress in near realtime. A few things needs to be done for this to happen.

#### Install clearml python on laptop
```bash
pip install clearml
```
#### Inject clearml code into your codes
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

If you are not following one of the above frameworks, and your code is in Python, you can use the manual approach.
Details of the manual approach can be found here. <br>
Tutorial:https://allegro.ai/clearml/docs/docs/tutorials/tutorial_explicit_reporting.html <br>
Examples: https://allegro.ai/clearml/docs/rst/examples/explicit_reporting/index.html
