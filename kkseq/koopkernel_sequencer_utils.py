"""Utils for Koopman Kernel Seq2Seq architecture."""

# from datetime import datetime
import logging

import numpy as np
import torch
from kooplearn.data import TensorContextDataset
from torch.optim import Optimizer

from kkseq.koopkernel_sequencer import (
    KoopKernelLoss,
    NystroemKoopKernelSequencer,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def RMSE(
    test_preds,
    test_tgts
):
  """Regular RMSE metric for TCTracks data.

  Args:
    test_preds: # models' predictions with shape of (number of trajectories,
    number of samples for traj, forecasting horizons, 2 velocity components)
    test_tgts: ground truth that has the same shape as test_preds.

  Returns:
    short, medium, long forecasting horizon prediction Weighted RMSE.
  """
  fh = test_preds.shape[1]
  mse = np.mean((test_preds - test_tgts)**2, axis=0)
  # return short, medium, long forecasting horizon and total RMSE
  return np.sqrt(np.mean(mse[:fh // 3])), np.sqrt(
      np.mean(mse[fh // 3:fh // 3 * 2])), np.sqrt(
          np.mean(mse[fh // 3 * 2:])), np.sqrt(np.mean(mse))


def train_one_epoch(
    model: NystroemKoopKernelSequencer,
    optimizer: Optimizer,
    loss_fun: KoopKernelLoss,
    tensor_context_inps: torch.tensor,
    tensor_context_tgts: torch.tensor,
) -> float:
    """Train one epoch.

    Args:
        model (NystroemKoopKernelSequencer): _description_
        optimizer (Optimizer): _description_
        loss_fun (KoopKernelLoss): _description_
        tensor_context_inps (torch.tensor): Input tensor context with shape
            (batch_size, n_data // batch_size, input_length, num_feats).
        tensor_context_tgts (torch.tensor): Output tensor context with shape
            output_length = 1:
            (batch_size, n_data // batch_size, input_length, num_feats),
            output_length > 1:
            (batch_size, n_data // batch_size, input_length, output_length, num_feats).

    Returns:
        float: Square root of MSL.
    """
    train_loss = []
    for i in range(tensor_context_inps.shape[1]):
        # Every data instance is an input + label pair
        inputs, labels = tensor_context_inps[:, i], tensor_context_tgts[:, i]
        if model.context_mode == "last_context":
            # FIXME This might not work for output_length > 1.
            labels = labels[:, -1, :]

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss
        loss = loss_fun(outputs, labels)
        train_loss.append(loss.item())

        # Zero your gradients for every batch, and compute loss gradient.
        optimizer.zero_grad()
        loss.backward()

        # clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)

        # Adjust learning weights
        optimizer.step()

    return np.sqrt(np.mean(train_loss))


def eval_one_epoch(
    model: NystroemKoopKernelSequencer,
    loss_fun: KoopKernelLoss,
    tensor_context_inps: torch.tensor,
    tensor_context_tgts: torch.tensor,
):
    eval_loss = []
    all_preds = []
    all_trues = []
    for i in range(tensor_context_inps.shape[1]):
        # Every data instance is an input + label pair
        inputs, labels = tensor_context_inps[:, i], tensor_context_tgts[:, i]
        if model.context_mode == "last_context":
            labels = labels[:, -1, :]

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss
        loss = loss_fun(outputs, labels)
        eval_loss.append(loss.item())
        all_preds.append(outputs.cpu().data.numpy())
        all_trues.append(labels.cpu().data.numpy())

    return (
        np.sqrt(np.mean(eval_loss)),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_trues, axis=0),
    )


def train_one_epoch_old(
    model: NystroemKoopKernelSequencer,
    optimizer,
    loss_fun: KoopKernelLoss,
    epoch_index,
    tb_writer,
    tensor_context_inps,
    tensor_context_tgts,
):
    """From https://pytorch.org/tutorials/beginner/introyt/trainingyt.html."""

    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting

    for i in range(tensor_context_inps.shape[1]):
        # Every data instance is an input + label pair
        inputs, labels = tensor_context_inps[:, i], tensor_context_tgts[:, i]
        if model.context_mode == "last_context":
            labels = labels[:, -1, :]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fun(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * tensor_context_inps.shape[1] + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def predict_koopkernel_sequencer(
    time_series: torch.Tensor,
    prediction_steps: int,
    model: NystroemKoopKernelSequencer,
):
    assert len(time_series.shape) == 2
    # tgts_dummy only needed to set the number of predicted time steps by model.
    """
    Note: The model performs a number of single_forward steps, each predicting
    `num_steps` future time steps, such that the total number of predicted time steps is
    larger than `tgts.shape[1]`, i.e. `prediction_steps` or the `output_length` of the
    dataset. In the end only the first `output_length` predicted data points are given
    as output, the remaining (further) steps are discarded.
    """

    num_feats = time_series.shape[1]

    if model.context_mode == "no_context":
        time_series_unsqueeze = time_series.unsqueeze(0).to(device)
    elif model.context_mode == "full_context":
        time_series_unsqueeze = (
            time_series[-model.input_length :].unsqueeze(0).to(device)
        )
    if model.context_mode == "last_context":
        time_series_unsqueeze = (
            time_series[-model.input_length :].unsqueeze(0).to(device)
        )

    n_eval_steps = int(prediction_steps // model.output_length)
    if n_eval_steps * model.output_length < prediction_steps:
        n_eval_steps += 1

    predictions = torch.zeros(
        size=(
            n_eval_steps * model.output_length,
            num_feats,
        ),
        device=device,
        dtype=torch.float32,
    )
    prediction = time_series_unsqueeze
    # shape: (1, input_length, num_feats)

    for idx in range(n_eval_steps):
        new_prediction = model(prediction)
        if model.context_mode == "last_context":
            if model.output_length == 1:
                new_prediction = new_prediction.unsqueeze(1)
            prediction = torch.cat(
                [prediction[:, model.output_length :], new_prediction], dim=1
            )
            # shape: (1, input_length, num_feats)
        else:
            if model.output_length == 1:
                new_prediction = new_prediction.unsqueeze(2)
            prediction = torch.cat(
                [prediction[:, model.output_length :], new_prediction[:, -1]], dim=1
            )
            # shape: (1, input_length, num_feats)
        predictions[idx * model.output_length : (idx + 1) * model.output_length] = (
            prediction[0, -model.output_length :]
        )
    # shape: (n_eval_steps, num_feats)

    print(predictions.shape)

    return predictions[:prediction_steps]


def get_model_name(flag_params: dict) -> str:
    """Get model name.

    Args:
        flag_params (dict): _description_

    Raises:
        Exception: _description_

    Returns:
        str: _description_
    """
    model_name = "seed{}_bz{}_lr{}_decay{}_inp{}_pred{}_kknc{}_kkls{}_ctxm{}_mko{}_mv{}_ncw{}".format(  # noqa: E501, UP032
        flag_params["seed"],
        flag_params["batch_size"],
        flag_params["learning_rate"],
        flag_params["decay_rate"],
        flag_params["input_length"],
        flag_params["train_output_length"],
        flag_params["koopman_kernel_num_centers"],
        flag_params["koopman_kernel_length_scale"],
        flag_params["context_mode"],
        flag_params["mask_koopman_operator"],
        flag_params["mask_version"],
        flag_params["use_nystroem_context_window"],
    )

    return model_name