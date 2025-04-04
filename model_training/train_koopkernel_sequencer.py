"""Comparing KNF and koopkernel sequence models."""

import logging
import os
import random
import time
from datetime import datetime
from itertools import product

import numpy as np
from kooplearn.datasets import Lorenz63

from kkseq.data_utils import (
    LinearScaler,
)
from kkseq.koopkernel_sequencer import (
    NystroemKoopKernelSequencer,
    RBFKernel,
)
from kkseq.koopkernel_sequencer_utils import (
    RMSE,
    get_model_name,
)
from kkseq.train_koop_kernel_sequencer import train_KoopKernelSequencer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


time_lag = 1


# Generate data from Lorenz63 model.
configs = {
    "train_samples": 10000,
    "valid_samples": 1000,
    "test_samples": 100,
}
raw_data = Lorenz63().sample(
    X0=np.ones(3),
    T=configs["train_samples"]
    + 1000
    + configs["valid_samples"]
    + 1000
    + configs["test_samples"],
)
mean = np.mean(raw_data, axis=0)
norm = np.max(np.abs(raw_data), axis=0)
# Data rescaling
data = raw_data - mean
data /= norm
dataset = {
    "train": data[: configs["train_samples"] + 1],
    "valid": data[
        configs["train_samples"] + 1 + 1000 : configs["train_samples"]
        + 1
        + 1000
        + configs["valid_samples"]
        + 1
    ],
    "test": data[-configs["test_samples"] - 1 :],
}
time_series_list_train = [dataset["train"]]
time_series_list_valid = [dataset["valid"]]
time_series_list_test = [dataset["test"]]


# Set training settings
# training_settings = {
#     "koopman_kernel_length_scale": [0.24],
#     "koopman_kernel_num_centers": [100],
#     "context_mode": ["last_context"],
#     "mask_koopman_operator": [False],
#     "mask_version": [1],
#     "use_nystroem_context_window": [False],
#     "output_length": [1],
# }
training_settings = {
    "koopman_kernel_length_scale": [0.16, 0.18, 0.20, 0.22, 0.24],
    "koopman_kernel_num_centers": [500],
    # "koopman_kernel_num_centers": [1000],
    "context_mode": ["full_context", "last_context"],
    "mask_koopman_operator": [True, False],
    "mask_version": [1],
    "use_nystroem_context_window": [False, True],
    "output_length": [1],
}


flag_params = {}
flag_params["batch_size"] = 16
flag_params["input_length"] = 12
flag_params["learning_rate"] = 0.001
flag_params["decay_rate"] = 0.9

flag_params["num_epochs"] = 50
flag_params["seed"] = 24


random.seed(flag_params["seed"])  # python random generator
np.random.seed(flag_params["seed"])  # numpy random generator


# Logging and define save paths
current_file_dir_path = os.path.dirname(os.path.abspath(__file__))
# current_file_dir_path = os.getcwd()



flag_params["dataset"] = "lorenz63"

model_str = "koopkernelseq"
print("===================================")
print(f"Train {model_str}.")
print("===================================")
print()

date_time = datetime.fromtimestamp(time.time())
str_date_time = date_time.strftime("%Y-%m-%d-%H-%M-%S")
flag_params["model"] = model_str

results_dir = os.path.join(
    current_file_dir_path,
    "training_results",
    "kkseq",
    flag_params["model"],
    str_date_time,
)
logs_dir = os.path.join(
    current_file_dir_path,
    "logs",
    "kkseq",
    flag_params["model"],
)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

log_file_name = os.path.join(logs_dir, flag_params["model"] + ".log")

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    # style="{",
    # datefmt="%Y-%m-%d %H:%M:%S",
)
fileHandler = logging.FileHandler(log_file_name)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

for (
    koopman_kernel_length_scale,
    koopman_kernel_num_centers,
    context_mode,
    mask_koopman_operator,
    mask_version,
    use_nystroem_context_window,
    output_length,
) in product(*training_settings.values()):
    print()
    print()
    print("=============================================================")
    print("Iteration:")
    print(
        koopman_kernel_length_scale,
        koopman_kernel_num_centers,
        context_mode,
        mask_koopman_operator,
        mask_version,
        use_nystroem_context_window,
        output_length,
    )
    # if not mask_koopman_operator:
    #     if mask_version == 0:
    #         print("Skip iteration.")
    #         continue
    if context_mode == "last_context":
        if mask_koopman_operator:
            print("Skip iteration.")
            continue
        # if mask_version == 0:
        #     print("Skip iteration.")
        #     continue

    flag_params["train_output_length"] = output_length
    flag_params["test_output_length"] = flag_params["train_output_length"]

    flag_params["koopman_kernel_length_scale"] = koopman_kernel_length_scale
    flag_params["koopman_kernel_num_centers"] = koopman_kernel_num_centers
    flag_params["context_mode"] = context_mode
    flag_params["mask_koopman_operator"] = mask_koopman_operator
    flag_params["mask_version"] = mask_version
    flag_params["use_nystroem_context_window"] = use_nystroem_context_window
    if flag_params["context_mode"] == "no_context":
        flag_params["input_length"] = (
            4  # small input_length for context_mode = no_context
        )
    else:
        flag_params["input_length"] = 12
    flag_params["context_length"] = (
        flag_params["input_length"] + flag_params["train_output_length"]
    )

    logger.info(flag_params)

    # Model training.
    scaler = LinearScaler()

    eval_metric = RMSE

    model_name = get_model_name(flag_params)

    rbf = RBFKernel(length_scale=flag_params["koopman_kernel_length_scale"])
    koopkernelmodel = NystroemKoopKernelSequencer(
        kernel=rbf,
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
        output_dim=1,
        num_nys_centers=flag_params["koopman_kernel_num_centers"],
        rng_seed=42,
        context_mode=flag_params["context_mode"],
        mask_koopman_operator=flag_params["mask_koopman_operator"],
        mask_version=flag_params["mask_version"],
        use_nystroem_context_window=flag_params["use_nystroem_context_window"],
    )

    model, all_train_rmses = train_KoopKernelSequencer(
        model=koopkernelmodel,
        eval_metric=eval_metric,
        time_series_list_train=time_series_list_train,
        time_series_list_valid=time_series_list_valid,
        time_series_list_test=time_series_list_test,
        num_epochs=flag_params["num_epochs"],
        batch_size=flag_params["batch_size"],
        scaler=scaler,
        flag_params=flag_params,
        results_dir=results_dir,
        model_name=model_name,
        save_model=False,
    )
