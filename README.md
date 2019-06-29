CycleGAN-Tensorflow
==============================

A Tensorflow implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) using Eager Execution, tf.keras.layers, and tf.data.

Requirements:

- Tensorflow 1.11
- Python 3.6

Project Organization
------------

    ├── README.md
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── LICENSE
    ├── eval_cityscapes    <- This contains scripts for getting the experiment results.
    └── src                <- Source code for use in this project
        ├── __init__.py    <- Makes src a Python module
        │
        ├── train.py       <- Run this to train
        │
        ├── test.py        <- Run this to test
        │
        ├── data           <- Code for downloading or loading data  
        │   ├── data.py         <- Dataset class
        │   └── download_data.py
        │
        ├── models         <- Code for defining the network structure and loss functions
        │   ├── cyclegan.py     <- CycleGAN model class
        │   ├── networks.py
        │   └── losses.py
        │
        └── utils          <- Utility files
            ├── options.py      <- Class for command line options
            └── image_history_buffer.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
