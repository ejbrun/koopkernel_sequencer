"""Comparing KNF and koopkernel sequence models."""

import logging
import random
import time
from itertools import product

import numpy as np
import torch
from kooplearn.datasets import Lorenz63

from kkseq.data_utils import (
    LinearScaler,
    standardized_batched_context_from_time_series_list,
)
from kkseq.koopkernel_sequencer import (
    KoopKernelLoss,
    NystroemKoopKernelSequencer,
    RBFKernel,
)
from kkseq.koopkernel_sequencer_utils import (
    RMSE,
    eval_one_epoch,
    train_one_epoch,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


time_lag = 1


configs = {
    "train_samples": 10000,
    "test_samples": 100,
}


raw_data = Lorenz63().sample(
    X0=np.ones(3), T=configs["train_samples"] + 1000 + configs["test_samples"]
)
mean = np.mean(raw_data, axis=0)
norm = np.max(np.abs(raw_data), axis=0)
# Data rescaling
data = raw_data - mean
data /= norm

dataset = {
    "train": data[: configs["train_samples"] + 1],
    "test": data[-configs["test_samples"] - 1 :],
}

time_series_list_train = [dataset["train"]]
time_series_list_test = [dataset["test"]]


flag_params = {}
flag_params["batch_size"] = 16
flag_params["koopman_kernel_length_scale"] = 2.5
flag_params["koopman_kernel_num_centers"] = 100
flag_params["mask_koopman_operator"] = False
flag_params["mask_version"] = 0
flag_params["use_nystroem_context_window"] = False

flag_params["learning_rate"] = 0.001
flag_params["decay_rate"] = 0.9

flag_params["num_epochs"] = 5
flag_params["seed"] = 41

random.seed(flag_params["seed"])  # python random generator
np.random.seed(flag_params["seed"])  # numpy random generator


test_setting = {
    "context_mode": ["full_context", "last_context"],
    "mask_koopman_operator": [True, False],
    "mask_version": [0, 1],
    "use_nystroem_context_window": [True, False],
    "output_length": [1, 3],
}

for (
    context_mode,
    mask_koopman_operator,
    mask_version,
    use_nystroem_context_window,
    output_length,
) in product(*test_setting.values()):
    print("=================")
    print(
        context_mode,
        mask_koopman_operator,
        mask_version,
        use_nystroem_context_window,
        output_length,
    )

    flag_params["train_output_length"] = output_length
    flag_params["test_output_length"] = flag_params["train_output_length"]

    flag_params["context_mode"] = context_mode
    if flag_params["context_mode"] == "no_context":
        flag_params["input_length"] = (
            4  # small input_length for context_mode = no_context
        )
    else:
        flag_params["input_length"] = 12
    flag_params["context_length"] = (
        flag_params["input_length"] + flag_params["train_output_length"]
    )

    scaler = LinearScaler()
    eval_metric = RMSE
    model_name = "test_model"

    rbf = RBFKernel(length_scale=flag_params["koopman_kernel_length_scale"])
    koopkernelmodel = NystroemKoopKernelSequencer(
        kernel=rbf,
        input_length=flag_params["input_length"],
        output_length=flag_params["train_output_length"],
        output_dim=1,
        num_nys_centers=flag_params["koopman_kernel_num_centers"],
        rng_seed=flag_params["seed"],
        context_mode=flag_params["context_mode"],
        mask_koopman_operator=flag_params["mask_koopman_operator"],
        mask_version=flag_params["mask_version"],
        use_nystroem_context_window=flag_params["use_nystroem_context_window"],
    )

    total_params = sum(p.numel() for p in koopkernelmodel.parameters())
    print(f"Number of parameters: {total_params}")

    # Validation and test sets are the same.
    time_series_list_valid = time_series_list_test
    tensor_context_inps_train, tensor_context_tgts_train = (
        standardized_batched_context_from_time_series_list(
            time_series_list_train,
            flag_params["batch_size"],
            scaler,
            context_length=flag_params["context_length"],
            time_lag=time_lag,
            fit=True,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
        )
    )
    # shapes: (batch_size, n_data // batch_size, input_length, num_feats) or
    # (batch_size, n_data // batch_size, input_length, output_length, num_feats)
    tensor_context_inps_valid, tensor_context_tgts_valid = (
        standardized_batched_context_from_time_series_list(
            time_series_list_valid,
            flag_params["batch_size"],
            scaler,
            context_length=flag_params["context_length"],
            time_lag=time_lag,
            fit=False,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
        )
    )
    tensor_context_inps_test, tensor_context_tgts_test = (
        standardized_batched_context_from_time_series_list(
            time_series_list_test,
            flag_params["batch_size"],
            scaler,
            context_length=flag_params["context_length"],
            time_lag=time_lag,
            fit=False,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
        )
    )

    koopkernelmodel._initialize_nystrom_data(
        tensor_context_inps_train=tensor_context_inps_train,
        tensor_context_tgts_train=tensor_context_tgts_train,
    )

    optimizer = torch.optim.Adam(
        koopkernelmodel.parameters(), lr=flag_params["learning_rate"]
    )
    loss_koopkernel = KoopKernelLoss(
        koopkernelmodel.nystrom_data_Y, koopkernelmodel._kernel
    )

    t_stamp = time.time()
    for _, _ in enumerate(range(flag_params["num_epochs"])):
        # print(epoch_index)
        start_time = time.time()

        train_rmse = train_one_epoch(
            koopkernelmodel,
            optimizer,
            loss_koopkernel,
            tensor_context_inps_train,
            tensor_context_tgts_train,
        )
        eval_rmse, _, _ = eval_one_epoch(
            koopkernelmodel,
            loss_koopkernel,
            tensor_context_inps_valid,
            tensor_context_tgts_valid,
        )

    print("Runtime:", time.time() - t_stamp)
