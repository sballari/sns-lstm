"""Module that defines the function that load the hyperparamters from a config
file.

"""
import yaml


def load_hparams(fparams):
    """Returns a dictionary with the loaded hyperparameters.

    Args:
      fparams: string. Filepath to the hyperparameter file.

    """
    with open(fparams, "r") as fp:
        return yaml.safe_load(fp)
