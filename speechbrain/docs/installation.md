
# Quick installation

SpeechBrain is constantly evolving. New features, tutorials, and documentation will appear over time. SpeechBrain can be installed via PyPI to rapidly use the standard library. Moreover, a local installation can be used to run experiments and modify/customize the toolkit and its recipes.

SpeechBrain supports both CPU and GPU computation. For most recipes, however, a GPU is necessary during training. Please note that CUDA must be properly installed to use GPUs.

We support [PyTorch](https://pytorch.org/get-started/locally/) 1.9+ and Python 3.8-3.12 (newer Python versions than advertised may work if supported by PyTorch).
We recommend you upgrade to at least 3.9+ as support for 3.8 will be removed eventually.

## Install via PyPI

Once you have created your Python environment (see instructions below) you can simply type:

```
pip install speechbrain
```

Depending on your OS, audio loading may require the install of optional torchaudio dependencies to work. If it does not work out-of-the box for you, please visit [audio troubleshooting](audioloading.html).

Then you can then access SpeechBrain with:

```
import speechbrain as sb
```

## Install locally

Once you have created your Python environment (see instructions below) you can simply type:

```
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```

Then you can access SpeechBrain with:

```
import speechbrain as sb
```

Any modification made to the `speechbrain` package will be automatically interpreted as we installed it with the `--editable` flag.

## Test installation
Please, run the following script  from the main folder to make sure your installation is working:
```
pytest tests
```

If you have a GPU, you can run the tests with `pytest tests --device='cuda'`

You can run doctests with:

```
tests/.run-doctests.sh
```

## Operating Systems

SpeechBrain supports Linux-based distributions and macOS. A solution for windows users can be found
in this [GitHub issue](https://github.com/speechbrain/speechbrain/issues/512).

## Setting up a Conda environment/virtualenv

A good practice is to have different python environments for your different tools
and toolkits, so they do not interfere with each other. This can be done either with
[Anaconda](https://www.anaconda.com/products/distribution) or [venv](https://docs.python.org/3.8/library/venv.html).

Anaconda can be installed by simply following [this tutorial](https://docs.anaconda.com/free/anaconda/install/linux/). In practice, it is a matter of downloading the installation script and executing it.

### Conda

Once Conda is installed, you can create a new environment with:

```
conda create --name speechbrain python=3.11
```

Then, activate it with:

```
conda activate speechbrain
```

Now, you can install all the needed packages!

More information on managing environments with Anaconda can be found in [the documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## venv setup

venv is even simpler. To create your environment:

```
python3 -m venv /path/to/new/virtual/speechbrain
```

And to activate it:

```
source /path/to/new/virtual/speechbrain/bin/activate
```

Now, you can install all the needed packages!



## Test your GPU installation

As SpeechBrain only relies on PyTorch, its GPU usage is also linked to it. Hence,
if PyTorch sees your GPUs, SpeechBrain will. Many functions can be called from the `torch` package to verify that your GPUs are detected:

```
import torch

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)
```
