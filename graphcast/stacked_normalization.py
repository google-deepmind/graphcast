"""Wrappers for Predictors which allow them to work with normalized data.

The Predictor which is wrapped sees normalized inputs and targets, and makes
normalized predictions. The wrapper handles translating the predictions back
to the original domain.
"""

import logging
import chex
from typing import Optional, Tuple

from graphcast.stacked_predictor_base import StackedPredictor, StackedLossAndDiagnostics
from graphcast import xarray_tree
import xarray


def normalize(values: chex.Array,
              scales: chex.Array,
              locations: Optional[chex.Array],
              ) -> chex.Array:
    """Normalize variables using the given scales and (optionally) locations."""
    result = values
    if locations is not None:
        result -= locations.astype(values.dtype)

    result /= scales.astype(values.dtype)
    return result


def unnormalize(values: chex.Array,
                scales: chex.Array,
                locations: Optional[chex.Array],
                ) -> chex.Array:
    """Unnormalize variables using the given scales and (optionally) locations."""
    result = values * scales#.astype(values.dtype)

    if locations is not None:
        result += locations#.astype(values.dtype)
    return result


class StackedInputsAndResiduals(StackedPredictor):
    """Wraps with a residual connection, normalizing inputs and target residuals.

    The inner predictor is given inputs that are normalized using `locations`
    and `scales` to roughly zero-mean unit variance.

    For target variables that are present in the inputs, the inner predictor is
    trained to predict residuals (target - last_frame_of_input) that have been
    normalized using `residual_scales` (and optionally `residual_locations`) to
    roughly unit variance / zero mean.

    This replaces `residual.Predictor` in the case where you want normalization
    that's based on the scales of the residuals.

    Since we return the underlying predictor's loss on the normalized residuals,
    if the underlying predictor is a sum of per-variable losses, the normalization
    will affect the relative weighting of the per-variable loss terms (hopefully
    in a good way).

    For target variables *not* present in the inputs, the inner predictor is
    trained to predict targets directly, that have been normalized in the same
    way as the inputs.

    The transforms applied to the targets (the residual connection and the
    normalization) are applied in reverse to the predictions before returning
    them.
    """

    def __init__(
        self,
        predictor: StackedPredictor,
        stddev_by_level: dict[chex.Array, chex.Array],
        mean_by_level: dict[chex.Array, chex.Array],
        diffs_stddev_by_level: dict[chex.Array, chex.Array],
        last_input_channel_mapping: dict,
        ):
        self._predictor = predictor
        self._scales = stddev_by_level
        self._locations = mean_by_level
        self._residual_scales = diffs_stddev_by_level
        self._residual_locations = {"inputs": None, "targets": None}
        self._last_input_channel_mapping = last_input_channel_mapping

        self._checkit(self._scales)
        self._checkit(self._locations)
        self._checkit(self._residual_scales)

    @staticmethod
    def _checkit(attr):
        if attr is not None:
            assert isinstance(attr, dict)
            assert "inputs" in attr.keys()
            assert "targets" in attr.keys()
            assert len(attr.keys()) == 2


    def _unnormalize_prediction_and_add_input(self, inputs, norm_prediction):
        """
        arrays have shape [lat, lon, batch, time, channels]
        Note:
            Assumes that all prediction variables are also inputs!!
        """

        # grab the predictions that are also inputs, normalize these with residuals
        prediction = unnormalize(
            norm_prediction,
            self._residual_scales["targets"],
            self._residual_locations["targets"],
        )
        last_input = inputs[..., list(self._last_input_channel_mapping.values())]
        prediction += last_input
        return prediction

    def _subtract_input_and_normalize_target(self, inputs, target):
        last_input = inputs[..., list(self._last_input_channel_mapping.values())]
        target_residual = target - last_input
        result = normalize(
            target_residual,
            self._residual_scales["targets"],
            self._residual_locations["targets"],
        )
        return result

    def __call__(
        self,
        inputs: chex.Array,
        **kwargs
        ) -> chex.Array:
        norm_inputs = normalize(inputs, self._scales["inputs"], self._locations["inputs"])
        norm_predictions = self._predictor(norm_inputs, **kwargs)
        return self._unnormalize_prediction_and_add_input(inputs, norm_predictions)

    def loss(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        **kwargs,
        ) -> StackedLossAndDiagnostics:
        """Returns the loss computed on normalized inputs and targets."""
        norm_inputs = normalize(inputs, self._scales["inputs"], self._locations["inputs"])
        norm_target_residuals = self._subtract_input_and_normalize_target(inputs, targets)
        return self._predictor.loss(norm_inputs, norm_target_residuals, **kwargs)

    def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
        self,
        inputs: chex.Array,
        targets: chex.Array,
        **kwargs,
        ) -> Tuple[StackedLossAndDiagnostics, chex.Array]:
        """Returns the loss computed on normalized inputs and targets."""
        norm_inputs = normalize(inputs, self._scales["inputs"], self._locations["inputs"])
        norm_target_residuals = self._subtract_input_and_normalize_target(inputs, targets)
        (loss, scalars), norm_predictions = self._predictor.loss_and_predictions(
            norm_inputs,
            norm_target_residuals,
            **kwargs,
        )
        predictions = self._unnormalize_prediction_and_add_input(inputs, norm_predictions)
        return (loss, scalars), predictions
