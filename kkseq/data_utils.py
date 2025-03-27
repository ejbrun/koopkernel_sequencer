"""Utils for data handling."""

import logging

import numpy as np
from numpy.typing import NDArray
import torch
from kooplearn.data import TensorContextDataset, TrajectoryContextDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearScalerError(Exception):
    """A custom exception used to report errors in use of Timer class."""


class LinearScaler:
    """MinMaxScaler von sklearn is very similar.

    However, MinMaxScaler does not allow to scale different dimensions to different
    intervals. This functionality might be useful down the road.
    """

    def __init__(self, target_min_vec=None, target_max_vec=None):
        """Initialization.

        Args:
            target_min_vec (_type_, optional): _description_. Defaults to None.
            target_max_vec (_type_, optional): _description_. Defaults to None.
        """
        self.target_min_vec = target_min_vec
        self.target_max_vec = target_max_vec
        self.min_vec = None
        self.max_vec = None

    def _linear_transform(self, data):
        scaling_factor = (self.target_max_vec - self.target_min_vec) / (
            self.max_vec - self.min_vec
        )
        diffs_to_min = data - self.min_vec
        scaled_diffs_to_min_vec = scaling_factor * diffs_to_min
        return scaled_diffs_to_min_vec + self.target_min_vec

    def _linear_transform_2(
        self, data, input_min_vec, input_max_vec, target_min_vec, target_max_vec
    ):
        scaling_factor = (target_max_vec - target_min_vec) / (
            input_max_vec - input_min_vec
        )
        diffs_to_min = data - input_min_vec
        scaled_diffs_to_min_vec = scaling_factor * diffs_to_min
        return scaled_diffs_to_min_vec + target_min_vec

    def transform(self, data):
        """Transform.

        Args:
            data (_type_): _description_

        Raises:
            LinearScalerError: _description_

        Returns:
            _type_: _description_
        """
        if self.min_vec is None or self.max_vec is None:
            raise LinearScalerError(
                "Cannot transform. You first have to call .fit_transform()."
            )

        return self._linear_transform(data)

    def inverse_transform(self, data):
        """Inverse transform.

        Args:
            data (_type_): _description_

        Raises:
            LinearScalerError: _description_

        Returns:
            _type_: _description_
        """
        if self.min_vec is None or self.max_vec is None:
            raise LinearScalerError(
                "Cannot inverse-transform. You first have to call .fit_transform()."
            )
        inverse_transformed_data = self._linear_transform_2(
            data,
            input_min_vec=self.target_min_vec,
            input_max_vec=self.target_max_vec,
            target_min_vec=self.min_vec,
            target_max_vec=self.max_vec,
        )
        return inverse_transformed_data

    def fit_transform(self, data):
        """Fit transform.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.target_min_vec is None:
            self.target_min_vec = np.array([-1.0 for _ in range(data.shape[-1])])
        if self.target_max_vec is None:
            self.target_max_vec = np.array([1.0 for _ in range(data.shape[-1])])
        self.max_vec = np.max(data, axis=0)
        self.min_vec = np.min(data, axis=0)
        return self.transform(data)


def context_dataset_from_time_series_list(
    time_series_list: list[NDArray],
    context_length: int = 2,
    time_lag: int = 1,
    backend: str = "auto",
    verbose: int = 0,
    **backend_kw,
) -> TensorContextDataset:
    """Generate context dataset from TCTRacks.

    Args:
        time_series_list (list[NDArray]): List of time series. The zeroth axis of each
            time series is the time axis. Time series can be tensor-valued.
        context_length (int, optional): Length of the context window. Default to ``2``.
        time_lag (int, optional): Time lag, i.e. stride, between successive context
            windows. Default to ``1``.
        backend (str, optional): Specifies the backend to be used
            (``'numpy'``, ``'torch'``). If set to ``'auto'``, will use the same backend
            of the trajectory. Default to ``'auto'``.
        verbose (int, optional): Sets verbosity level. Default to 0 (no output).
        **backend_kw (dict, optional): Keyword arguments to pass to the backend.
            For example, if ``'torch'``, it is possible to specify the device of the
            tensor.

    Returns:
        TensorContextDataset: _description_
    """
    time_series = time_series_list[0]
    if len(time_series.shape) == 1:
        time_series = time_series[:, np.newaxis]
    
    context_data_array = np.empty((0, context_length, *time_series.shape[1:]))
    for idx, time_series in enumerate(time_series_list):
        if time_series.shape[0] > context_length * time_lag:
            if len(time_series.shape) == 1:
                time_series = time_series[:, np.newaxis]
            
            context_data_array = np.concatenate(
                [
                    context_data_array,
                    TrajectoryContextDataset(
                        time_series, context_length, time_lag, backend, **backend_kw
                    ).data,
                ],
                axis=0,
            )
        else:
            if verbose > 0:
                print(
                    f"""Data entry {idx} has been removed since it is shorter than the 
                    context_length {context_length} times time_lag {time_lag}."""
                )

    tensor_context_dataset = TensorContextDataset(
        context_data_array, backend, **backend_kw
    )
    return tensor_context_dataset


def standardize_TensorContextDataset(
    tensor_context: TensorContextDataset,
    scaler: StandardScaler | MinMaxScaler | LinearScaler,
    fit: bool = True,
    backend: str = "auto",
    **backend_kw,
) -> TensorContextDataset:
    """Standardizes a TensorContextDataset.

    Data standardization is performed by the scaler. Often used scalers are the
    standard scaler, which scales each feature to zero mean and unit variance, or global
    linear scaler, which transform the data by a affine linear transformation to a
    target rectangular domain.

    TODO At the moment the TensorContextDataset is standardized, by flattening into an
    array of shape (-1, n_features). An alternative would be to standardize the
    data_array_list (output of data_array_list_from_TCTracks), by concatenating all
    arrays of shape (-1, n_features) in data_array_list. This latter approach is
    implemented for the standardization of the KNF-adjusted dataset. The disadvantage
    of the former approach is that there might be some bias induced towards data points
    in the middle of the time series contained in data_array_list. This is because all
    these time series are sampled with a slicing window to homogenize the format into a
    dense array of time series of equal length, which is needed as input to the models.

    Args:
        tensor_context (TensorContextDataset): _description_
        scaler (StandardScaler | MinMaxScaler | LinearScaler): _description_
        fit (bool, optional): _description_. Defaults to True.
        backend (str, optional): _description_. Defaults to "auto".
        **backend_kw (dict, optional): Keyword arguments to pass to the backend.
                For example, if ``'torch'``, it is possible to specify the device of the
                tensor.

    Returns:
        TensorContextDataset: _description_
    """
    if fit:
        data_transformed = scaler.fit_transform(
            tensor_context.data.reshape(
                (
                    tensor_context.shape[0] * tensor_context.shape[1],
                    tensor_context.shape[2],
                )
            )
        ).reshape(tensor_context.shape)
    else:
        data_transformed = scaler.transform(
            tensor_context.data.reshape(
                (
                    tensor_context.shape[0] * tensor_context.shape[1],
                    tensor_context.shape[2],
                )
            )
        ).reshape(tensor_context.shape)
    tensor_context_transformed = TensorContextDataset(
        data_transformed, backend, **backend_kw
    )
    return tensor_context_transformed


def standardized_context_dataset_from_time_series_list(
    time_series_list: list[NDArray],
    scaler: StandardScaler | MinMaxScaler | LinearScaler,
    context_length: int = 2,
    time_lag: int = 1,
    fit: bool = True,
    verbose: int = 0,
    input_length: int | None = None,
    output_length: int | None = None,
    backend: str = "auto",
    **backend_kw,
) -> TensorContextDataset:
    """Generates standardized TensorContextDataset from TCTracks.

    Args:
        time_series_list (list[NDArray]): List of time series. The zeroth axis of each
            time series is the time axis. Time series can be tensor-valued. 
            TODO check tensor-valued
        tc_tracks (TCTracks): _description_
        scaler (StandardScaler | MinMaxScaler | LinearScaler): _description_
        context_length (int, optional): _description_. Defaults to 2.
        time_lag (int, optional): _description_. Defaults to 1.
        fit (bool, optional): _description_. Defaults to True.
        verbose (int, optional): _description_. Defaults to 1.
        input_length (int | None, optional): _description_. Defaults to None.
        output_length (int | None, optional): _description_. Defaults to None.
        backend (str, optional): _description_. Defaults to "auto".
        **backend_kw (dict, optional): Keyword arguments to pass to the backend.
                For example, if ``'torch'``, it is possible to specify the device of the
                tensor.
        
    Returns:
        TensorContextDataset: _description_
    """
    if input_length is not None and output_length is not None:
        con_len = input_length + output_length
    else:
        con_len = context_length
    
    tensor_context = context_dataset_from_time_series_list(
        time_series_list,
        context_length=con_len,
        time_lag=time_lag,
        backend=backend,
        verbose=verbose,
        **backend_kw
    )  
    standardized_tensor_context = standardize_TensorContextDataset(
        tensor_context,
        scaler=scaler,
        fit=fit,
        backend=backend,
        **backend_kw,
    )
    # shape: (n_data, input_length + output_length, num_feats)
    return standardized_tensor_context


def batch_tensor_context(
    tensor_context: TensorContextDataset,
    batch_size: int,
    input_length: int,
    output_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get batched tensor context.

    Args:
        tensor_context (TensorContextDataset): Tensor context of shape
            (n_data, input_length + output_length, num_feats).
        batch_size (int): Batch size.
        input_length (int): Input length.
        output_length (int): Output length.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of batched input and output tensor
            contexts, with shapes
            output_length = 1:
            (batch_size, n_data // batch_size, input_length, num_feats) and
            (batch_size, n_data // batch_size, input_length, num_feats).
            output_length > 1:
            (batch_size, n_data // batch_size, input_length, num_feats) and
            (batch_size, n_data // batch_size, input_length, output_length, num_feats).
    """
    if tensor_context.context_length != input_length + output_length:
        raise Exception(
            f"""
            tensor_context.context_lenght (={tensor_context.context_length}) must be
            equal to input_length (={input_length}) + output_length (={output_length}).
            """
        )
    if output_length == 1:
        tensor_context_inps = torch.tensor(
            tensor_context.lookback(input_length), dtype=torch.float32
        ).to(device)
        # shape: (n_data, input_length, num_feats)
        tensor_context_tgts = torch.tensor(
            tensor_context.lookback(input_length, slide_by=1),
            dtype=torch.float32,
        ).to(device)
        # shape: (n_data, input_length, num_feats)
    else:
        tensor_context_inps = torch.tensor(
            tensor_context.lookback(input_length), dtype=torch.float32
        ).to(device)
        # shape: (n_data, input_length, num_feats)
        tensor_context_tgts = torch.tensor(
            np.array(
                [
                    tensor_context.lookback(input_length, slide_by=idx + 1)
                    for idx in range(output_length)
                ]
            ),
            dtype=torch.float32,
        ).to(device)
        # shape: (output_length, n_data, input_length, num_feats)
        tensor_context_tgts = torch.einsum("abcd->bcad", tensor_context_tgts)
        # shape: (n_data, input_length, output_length, num_feats)

    # FIXME add random seed to randperm.
    rand_perm = torch.randperm(tensor_context_inps.shape[0])
    integer_divisor = tensor_context_inps.shape[0] // batch_size

    tensor_context_inps = tensor_context_inps[rand_perm]
    tensor_context_tgts = tensor_context_tgts[rand_perm]

    tensor_context_inps = tensor_context_inps[: integer_divisor * batch_size]
    tensor_context_tgts = tensor_context_tgts[: integer_divisor * batch_size]

    tensor_context_inps = tensor_context_inps.reshape(
        shape=[
            batch_size,
            integer_divisor,
            *tensor_context_inps.shape[1:],
        ]
    )
    # shape: (batch_size, n_data // batch_size, input_length, num_feats)
    tensor_context_tgts = tensor_context_tgts.reshape(
        shape=[
            batch_size,
            integer_divisor,
            *tensor_context_tgts.shape[1:],
        ]
    )
    # shape: (batch_size, n_data // batch_size, input_length, output_length, num_feats)

    return tensor_context_inps, tensor_context_tgts


def standardized_batched_context_from_time_series_list(
    time_series_list: list[NDArray],
    batch_size: int,
    scaler: StandardScaler | MinMaxScaler | LinearScaler,
    context_length: int = 2,
    time_lag: int = 1,
    fit: bool = True,
    verbose: int = 0,
    input_length: int | None = None,
    output_length: int | None = None,
    backend: str = "auto",
    **backend_kw,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate standardized and batched tensor contexts for inputs and outputs.

    Args:
        time_series_list (list[NDArray]): List of time series. The zeroth axis of each
            time series is the time axis. Time series can be tensor-valued.
        batch_size (int): _description_
        scaler (StandardScaler | MinMaxScaler | LinearScaler): _description_
        context_length (int, optional): _description_. Defaults to 2.
        time_lag (int, optional): _description_. Defaults to 1.
        fit (bool, optional): _description_. Defaults to True.
        verbose (int, optional): _description_. Defaults to 0.
        input_length (int | None, optional): _description_. Defaults to None.
        output_length (int | None, optional): _description_. Defaults to None.
        backend (str, optional): _description_. Defaults to "auto".
        **backend_kw (dict, optional): Keyword arguments to pass to the backend.
                For example, if ``'torch'``, it is possible to specify the device of the
                tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of batched input and output tensor
            contexts, with shapes
            output_length = 1:
            (batch_size, n_data // batch_size, input_length, num_feats) and
            (batch_size, n_data // batch_size, input_length, num_feats).
            output_length > 1:
            (batch_size, n_data // batch_size, input_length, num_feats) and
            (batch_size, n_data // batch_size, input_length, output_length, num_feats).
    """
    tensor_context = standardized_context_dataset_from_time_series_list(
        time_series_list,
        scaler,
        context_length,
        time_lag,
        fit,
        verbose,
        input_length,
        output_length,
        backend,
        **backend_kw,
    )
    # shape: (n_data, input_length + output_length, num_feats)
    tensor_context_inps, tensor_context_tgts = batch_tensor_context(
        tensor_context,
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
    )
    # shapes: (batch_size, n_data // batch_size, input_length, num_feats) or
    # (batch_size, n_data // batch_size, input_length, output_length, num_feats)
    return tensor_context_inps, tensor_context_tgts
