import os
import sys
import torch
import logging
import traceback
import numpy as np
from pprint import pprint

from runner import *
from utils.logger import setup_logging
from utils.arg_helper import parse_arguments, get_config

torch.set_printoptions(profile='full')

import pandas as pd

df = pd.read_csv("save_model_learning.csv")


def get_stats_from_trained_model(config_param, seed):
    """Return all mmd statistical results from
    generated graph by the trained model, in the form of a dict"""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config_param.use_gpu = config_param.use_gpu and torch.cuda.is_available()
    torch.cuda.empty_cache()

    runner = eval(config_param.runner)(config_param)

    mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = runner.test()
    return {"mmd_degree_test": mmd_degree_test, "mmd_clustering_test": mmd_clustering_test,
            "mmd_4orbits_test": mmd_4orbits_test, "mmd_spectral_test": mmd_spectral_test,
            "mmd_degree_dev": mmd_degree_dev, "mmd_clustering_dev": mmd_clustering_dev,
            "mmd_4orbits_dev": mmd_4orbits_dev, "mmd_spectral_dev": mmd_spectral_dev}


# %%

row_list = []
for training_path in df['file_dir']:

    try:
        config_path = os.path.join(training_path, 'config.yaml')
        config = get_config(config_path)
    except:
        continue

    for i in range(10):
        if training_path.find('mlp') == -1:
            dict_results = {"dataset_name": config.dataset.name, "model_name": config.model.name,
                            "num_epochs": config.train.max_epoch}
        else:
            dict_results = {"dataset_name": config.dataset.name, "model_name": config.model.name + "_MLP",
                            "num_epochs": config.train.max_epoch}
        dict_stats = get_stats_from_trained_model(config, 11 * (i ^ 3))
        dict_results.update(dict_stats)
        row_list.append(dict_results)
        torch.cuda.empty_cache()

result_df = pd.DataFrame(row_list)
torch.cuda.empty_cache()

result_df.to_csv("statsResults.csv")
result_df.groupby(['dataset_name', 'model_name', 'num_epochs']).agg(['mean', 'std']).to_csv("statssResults.csv")
