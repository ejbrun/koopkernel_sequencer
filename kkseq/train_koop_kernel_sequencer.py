"""Training of the Koopman Kernel Sequencer."""

# from datetime import datetime
import logging
import os

# import time
from time import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kkseq.data_utils import (
    LinearScaler,
    standardized_batched_context_from_time_series_list,
    standardized_context_dataset_from_time_series_list,
)
from kkseq.koopkernel_sequencer import (
    KoopKernelLoss,
    NystroemKoopKernelSequencer,
)
from kkseq.koopkernel_sequencer_utils import (
    RMSE,
    eval_one_epoch,
    train_one_epoch,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train_KoopKernelSequencer(
    model: NystroemKoopKernelSequencer,
    eval_metric: RMSE,
    time_series_list_train,
    time_series_list_test,
    num_epochs: int,
    batch_size: int,
    scaler: StandardScaler | MinMaxScaler | LinearScaler,
    flag_params: dict,
    results_dir: str | None = None,
    model_name: str | None = None,
    save_model: str = "best",
    split_valid_set: bool = True,
    early_stopping: bool = False,
    backend: str = "auto",
    **backend_kw,
) -> tuple[NystroemKoopKernelSequencer, list[float]]:
    """Train Koopman kernal sequence model.

    Args:
        model (NystroemKoopKernelSequencer): _description_
        eval_metric (RMSE_TCTracks): _description_
        tc_tracks (TCTracks | list[Dataset]): _description_
        num_epochs (int): _description_
        batch_size (int): _description_
        feature_list (_type_): _description_
        scaler (_type_): _description_
        basin (str): _description_
        input_length (int): _description_
        output_length (int): _description_
        decay_rate (float): _description_
        learning_rate (float): _description_
        log_file_handler (_type_): _description_
        results_dir (str): _description_
        model_name (str): _description_
        flag_params (dict): _description_
        save_model (str, optional): If model should be saved. For "best" only the best
            model is save, for "all" the model after each epoch is saved. For anything
            else, no model is saved. Defaults to "best".
        early_stopping (bool): If to apply early stopping. Defaults to False.
        backend (str, optional): _description_. Defaults to "auto".

    Raises:
        ValueError: _description_

    Returns:
        NystroemKoopKernelSequencer: _description_
    """
    if model_name is None:
        model_name = "default_model"

    if split_valid_set:
        time_series_list_train, time_series_list_valid = train_test_split(
            time_series_list_train, test_size=0.1, random_state=flag_params["seed"] + 1
        )
    else:
        time_series_list_valid = time_series_list_train

    tensor_context_train_standardized = (
        standardized_context_dataset_from_time_series_list(
            time_series_list_train,
            scaler=scaler,
            context_length=flag_params["context_length"],
            time_lag=1,
            fit=True,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
        )
    )

    # TODO first have to check _initialize for output_length > 1!!
    model._initialize_nystrom_data(tensor_context_train_standardized)
    del tensor_context_train_standardized

    tensor_context_inps_train, tensor_context_tgts_train = (
        standardized_batched_context_from_time_series_list(
            time_series_list_train,
            batch_size,
            scaler,
            context_length=flag_params["context_length"],
            time_lag=1,
            fit=True,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
            backend=backend,
            **backend_kw,
        )
    )
    # shapes: (batch_size, n_data // batch_size, input_length, num_feats) or
    # (batch_size, n_data // batch_size, input_length, output_length, num_feats)
    tensor_context_inps_valid, tensor_context_tgts_valid = (
        standardized_batched_context_from_time_series_list(
            time_series_list_valid,
            batch_size,
            scaler,
            context_length=flag_params["context_length"],
            time_lag=1,
            fit=False,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
            backend=backend,
            **backend_kw,
        )
    )
    tensor_context_inps_test, tensor_context_tgts_test = (
        standardized_batched_context_from_time_series_list(
            time_series_list_test,
            batch_size,
            scaler,
            context_length=flag_params["context_length"],
            time_lag=1,
            fit=False,
            input_length=flag_params["input_length"],
            output_length=flag_params["train_output_length"],
            backend=backend,
            **backend_kw,
        )
    )
    del time_series_list_train
    del time_series_list_valid
    del time_series_list_test

    if flag_params["train_output_length"] == 1:
        assert torch.all(
            tensor_context_inps_train[:, :, 1:] == tensor_context_tgts_train[:, :, :-1]
        )
    else:
        for idx in range(flag_params["train_output_length"]):
            assert torch.all(
                tensor_context_inps_train[:, :, idx + 1 :]
                == tensor_context_tgts_train[:, :, : -idx - 1, idx]
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=flag_params["learning_rate"])
    loss_koopkernel = KoopKernelLoss(model.nystrom_data_Y, model._kernel)

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # tb_writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=flag_params["decay_rate"]
    )  # stepwise learning rate decay

    all_train_rmses, all_eval_rmses = [], []
    best_eval_rmse = 1e6

    training_time_start = time()
    for epoch_index, epoch in enumerate(range(num_epochs)):
        start_time = time()

        train_rmse = train_one_epoch(
            model,
            optimizer,
            loss_koopkernel,
            tensor_context_inps_train,
            tensor_context_tgts_train,
        )
        eval_rmse, _, _ = eval_one_epoch(
            model,
            loss_koopkernel,
            tensor_context_inps_valid,
            tensor_context_tgts_valid,
        )

        print("eval comparison", eval_rmse, best_eval_rmse)
        if eval_rmse < best_eval_rmse:
            best_eval_rmse = eval_rmse
            best_model = model
        if results_dir is not None:
            if save_model == "all":
                torch.save(
                    [model, epoch, optimizer.param_groups[0]["lr"]],
                    os.path.join(results_dir, model_name + f"_epoch{epoch}" + ".pth"),
                )
            if save_model == "best":
                torch.save(
                    [best_model, epoch, optimizer.param_groups[0]["lr"]],
                    os.path.join(results_dir, model_name + "_best.pth"),
                )

        all_train_rmses.append(train_rmse)
        all_eval_rmses.append(eval_rmse)

        if np.isnan(train_rmse) or np.isnan(eval_rmse):
            raise ValueError("The model generate NaN values")

        # Save test scores.
        _, test_preds, test_tgts = eval_one_epoch(
            best_model,
            loss_koopkernel,
            tensor_context_inps_test,
            tensor_context_tgts_test,
        )
        if results_dir is not None:
            torch.save(
                {
                    "test_preds": test_preds,
                    "test_tgts": test_tgts,
                    "eval_score": eval_metric(
                        test_preds, test_tgts
                    ),  # FIXME eval_metric() here is a tuple of four elements, why? Should be a single number.
                },
                os.path.join(results_dir, f"ep{epoch}_test" + model_name + ".pt"),
            )
        # train the model at least 60 epochs and do early stopping
        if early_stopping:
            if epoch > flag_params["min_epochs"] and np.mean(
                all_eval_rmses[-10:]
            ) > np.mean(all_eval_rmses[-20:-10]):
                break

        epoch_time = time() - start_time
        scheduler.step()
        logger.info(
            "Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(  # noqa: UP032
                epoch + 1, epoch_time / 60, train_rmse, eval_rmse
            )
        )

    training_runtime = time() - training_time_start

    logger.info("Evaluate test metric.")
    _, test_preds, test_tgts = eval_one_epoch(
        best_model,
        loss_koopkernel,
        tensor_context_inps_test,
        tensor_context_tgts_test,
    )
    if results_dir is not None:
        torch.save(
            {
                "test_preds": test_preds,
                "test_tgts": test_tgts,
                "eval_score": eval_metric(
                    test_preds, test_tgts
                ),  # FIXME eval_metric() here is a tuple of four elements, why? Should be a single number.
                "train_rmses": all_train_rmses,
                "eval_rmses": all_eval_rmses,
                "training_runtime": training_runtime,
            },
            os.path.join(results_dir, "test_" + model_name + ".pt"),
        )
    # with open(os.path.join(results_dir, "test_" + model_name + ".json"), "w") as jsonfile:
    #     json.dump(
    #         {
    #             "eval_score": list(map(float, eval_metric(test_preds, test_tgts))), #FIXME eval_metric() here is a tuple of four elements, why? Should be a single number.
    #             "train_rmses": list(map(float, all_train_rmses)),
    #             "eval_rmses": list(map(float, all_eval_rmses)),
    #         },
    #         jsonfile,
    #         indent=4,
    #     )
    logger.info(f"eval_metric: {eval_metric(test_preds, test_tgts)}")

    return model, all_train_rmses
