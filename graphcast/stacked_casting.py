"""Wrappers that take care of casting."""

import contextlib
from typing import Any, Mapping, Tuple, Optional

import chex
from graphcast.stacked_predictor_base import StackedPredictor, StackedLossAndDiagnostics
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray

from graphcast.casting import (
    Bfloat16Cast,
    tree_map_cast,
    bfloat16_variable_view,
    _bfloat16_creator,
    _bfloat16_getter,
    _bfloat16_setter,
)


class StackedBfloat16Cast(Bfloat16Cast):
    """Wrapper that casts all inputs to bfloat16 and outputs to targets dtype."""

    def __init__(self, predictor: StackedPredictor, enabled: bool = True):
        """Inits the wrapper.

        Args:
          predictor: predictor being wrapped.
          enabled: disables the wrapper if False, for simpler hyperparameter scans.

        """
        self._enabled = enabled
        self._predictor = predictor

    def __call__(
        self,
        inputs: chex.Array,
        **kwargs
        ) -> chex.Array:
        if not self._enabled:
            return self._predictor(inputs, **kwargs)

        with bfloat16_variable_view():
            predictions = self._predictor(
                inputs.astype(jnp.bfloat16),
                **kwargs,
            )

        predictions_dtype = infer_floating_dtype(predictions)  # pytype: disable=wrong-arg-types
        if predictions_dtype != jnp.bfloat16:
            raise ValueError(f'Expected bfloat16 output, got {predictions_dtype}')

        targets_dtype = infer_floating_dtype(inputs)  # pytype: disable=wrong-arg-types
        return tree_map_cast(
            predictions,
            input_dtype=jnp.bfloat16,
            output_dtype=targets_dtype,
        )

    def loss(
        self,
        inputs: chex.Array,
        targets: chex.Array,
        weights: Optional[chex.Array | None] = None
        ) -> StackedLossAndDiagnostics:
        if not self._enabled:
            return self._predictor.loss(inputs, targets, weights)

        with bfloat16_variable_view():
            loss, scalars = self._predictor.loss(
                *_inputs_targets_weights_to_bfloat16(
                    inputs,
                    targets,
                    weights,
                )
            )

        if loss.dtype != jnp.bfloat16:
            raise ValueError(f'Expected bfloat16 loss, got {loss.dtype}')

        targets_dtype = infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types

        # Note that casting back the loss to e.g. float32 should not affect data
        # types of the backwards pass, because the first thing the backwards pass
        # should do is to go backwards the casting op and cast back to bfloat16
        # (and xprofs seem to confirm this).
        return tree_map_cast(
            (loss, scalars),
            input_dtype=jnp.bfloat16,
            output_dtype=targets_dtype,
        )

    def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
        self,
        inputs: chex.Array,
        targets: chex.Array,
        weights: Optional[chex.Array | None] = None
        ) -> Tuple[StackedLossAndDiagnostics,
                   chex.Array]:
        if not self._enabled:
            return self._predictor.loss_and_predictions(
                inputs,
                targets,
                weights,
            )

        with bfloat16_variable_view():
            (loss, scalars), predictions = self._predictor.loss_and_predictions(
                *_inputs_targets_weights_to_bfloat16(
                    inputs,
                    targets,
                    weights,
                )
            )

        if loss.dtype != jnp.bfloat16:
            raise ValueError(f'Expected bfloat16 loss, got {loss.dtype}')

        predictions_dtype = infer_floating_dtype(predictions)  # pytype: disable=wrong-arg-types
        if predictions_dtype != jnp.bfloat16:
            raise ValueError(f'Expected bfloat16 output, got {predictions_dtype}')

        targets_dtype = infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types
        return tree_map_cast(
            ((loss, scalars), predictions),
            input_dtype=jnp.bfloat16,
            output_dtype=targets_dtype,
        )


def infer_floating_dtype(array: chex.Array) -> np.dtype:
    """Infers a floating dtype from an input mapping of data."""
    return array.dtype if jnp.issubdtype(array.dtype, np.floating) else None

def _inputs_targets_weights_to_bfloat16(
    inputs: chex.Array,
    targets: chex.Array,
    weights: Optional[chex.Array | None] = None,
    ) -> Tuple[chex.Array,
               chex.Array,
               chex.Array]:

    i16 = inputs.astype(jnp.bfloat16)
    t16 = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), targets)
    w16 = weights.astype(jnp.bfloat16) if weights is not None else None
    return i16, t16, w16
