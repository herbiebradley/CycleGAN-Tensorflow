CycleGAN-Tensorflow
==============================

A Tensorflow implementation of [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) using Eager Execution, tf.keras.layers, and tf.data.

Requirements:

- Tensorflow 1.11.0
- Python 3.6

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md
    ├── data
    │   └── raw            <- Raw data before any processing.
    │
    ├── saved_models       <- Checkpointed models and tensorboard summaries.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── train.py       <- Run this to train.
        │
        ├── test.py        <- Run this to test.
        │
        ├── pipeline       <- Code for downloading or loading data  
        │   ├── data.py
        │   └── download_data.py
        │
        ├── options       <- Files for command line options
        │   └── base_options.py
        │
        ├── models         <- Code for defining the network structure and loss functions
        │   ├── __init__.py     <- model helper functions
        │   ├── network.py
        │   ├── cyclegan.py
        │   └── losses.py
        │
        └── utils  <- Utility files, including scripts for visualisation
            └── image_history_buffer.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
