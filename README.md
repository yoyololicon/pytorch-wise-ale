# Pytorch WiSE-ALE 
Implementation of [WiSE-ALE: Wide Sample Estimator for Approximate
Latent Embedding](https://arxiv.org/abs/1902.06160).
The project structure is brought from [pytorch-template](https://github.com/victoresque/pytorch-template).

## Quick start

An example config file `mnist.json`' is provided.

~~~~
{
    "name": "MNIST",
    "n_gpu": 1,

    "arch": {
        "type": "MNIST_VAE",
        "args": {}
    },
    "data_loader": {
        "type": "MnistDataLoader",
        "args":{
            "data_dir": "mnist_data/",
            "batch_size": 64,
            "shuffle": true,
            "validation_split": 0,
            "num_workers": 1
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.001
        }
    },
    "loss": "WiSE_UB2",     // Default loss function is WiSE-UB. Type 'AEVB' will use vanilla VAE objective.
    "metrics": [
        "kl_div", "reconstruct"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 60,
            "gamma": 0.5
        }
    },
    "trainer": {
        "epochs": 30,
        "save_dir": "mnist_saved/",
        "save_period": 10,
        "verbosity": 2,
        "monitor": "off",
        "early_stop": 0,

        "tensorboardX": true,
        "sample_size": 2
    }
}
~~~~
The setting is the same as in the paper appendix.

To train the model with example config:
```
python train.py -c mnist.json
```
The checkpoint files will be saved in `mnist_saved`. Other instructions please refer to pytorch-template.

## Example Files

* [mnist_visualization.ipynb](mnist_visualization.ipynb): A latent embedding visualization on mnist dataset.