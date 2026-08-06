# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constructors for dense layers (e.g. MLPs)."""

from collections.abc import Callable, Iterable, Mapping
import functools
from typing import Any, Optional, Required, TypedDict, Union
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import numpy as np


class DenseLayerKwargsExceptOutputSize(TypedDict, total=False):
  """Same as `DenseLayerKwargs` except for the output size."""

  hidden_size: Required[int]
  num_hidden_layers: Required[int]
  activation: Required[str]
  activation_normalization: str | None
  activate_final: bool
  with_bias: bool
  one_less_layer_when_activate_final: bool
  w_init: hk.initializers.Initializer | None
  b_init: hk.initializers.Initializer | None
  activation_normalization_kwargs: Mapping[str, Any] | None
  remat: bool = False  # pyrefly: ignore[bad-class-definition]
  stack: bool = False  # pyrefly: ignore[bad-class-definition]


class DenseLayerKwargs(DenseLayerKwargsExceptOutputSize, total=False):
  """Useful type for passing main configurable Kwargs of dense layers around.

  The optimizations arguments are left out of this, as this would typically
  be set at the call sites, rather than passing around.
  """

  output_size: Required[int]


class DenseLayer(hk.Module):
  """A dense layer with MLPs and normalization."""

  @hk.name_like("__call__")
  def __init__(
      self,
      *,
      name: str,
      output_size: int,
      hidden_size: int,
      num_hidden_layers: int,
      activation: str,
      activate_final: bool = False,
      one_less_layer_when_activate_final: bool = False,
      drop_first_matmul: bool = False,
      activation_normalization: Optional[str] = None,
      activation_normalization_kwargs: Optional[Mapping[str, Any]] = None,
      remat: bool = False,
      stack: bool = False,
      **mlp_kwargs,
  ):
    """Inits the module.

    Args:
      name: Name of the module.
      output_size: Size of the last MLP layer.
      hidden_size: Size of the MLP hidden leayers.
      num_hidden_layers: Number of hidden layers, the total number of layers
        will be `num_hidden_layers + 1`, unless
        `one_less_layer_when_activate_final and activate_final`.
      activation: Name of the activation function.
      activate_final: If True, the activation function will be applied to the
        last layer.
      one_less_layer_when_activate_final: If True, when `activate_final` is True
        the total number of layers will be `num_hidden_layers` instead of
        `num_hidden_layers + 1`.
      drop_first_matmul: If True, the first matmul will be dropped.
      activation_normalization: Type of normalization activation to use. One of
        "layer_norm", "rms_norm", "group_norm", to be passed to
        `NormalizationLayer`. None disables normalization.
      activation_normalization_kwargs: Other kwargs to be passed to
        `NormalizationLayer`.
      remat: Remat the DenseLayer.
      stack: If True, use stacked layers.
      **mlp_kwargs: additional kwargs for the MLP.
    """
    super().__init__(name=name)

    if activate_final and one_less_layer_when_activate_final:
      assert num_hidden_layers >= 1
      mlp_num_hidden_layers = num_hidden_layers - 1
    else:
      mlp_num_hidden_layers = num_hidden_layers
    output_sizes = [hidden_size] * mlp_num_hidden_layers + [output_size]
    self._mlp = MLP(
        output_sizes=output_sizes,
        name="mlp",
        activation=get_activation_fn(activation),
        activate_final=activate_final,
        drop_first_matmul=drop_first_matmul,
        stack=stack,
        **mlp_kwargs,)

    if activation_normalization is not None:
      self._normalization_layer = NormalizationLayer(
          name="normalization",
          activation_normalization=activation_normalization,
          stack=stack,
          **(activation_normalization_kwargs or {}),
      )
    else:
      self._normalization_layer = None

    self._remat = remat

  def __call__(
      self, inputs: jax.Array, norm_conditioning: jax.Array | None = None):
    def call_function(inputs: jax.Array, norm_conditioning: jax.Array | None):
      outputs = self._mlp(inputs)
      if self._normalization_layer is not None:
        outputs = self._normalization_layer(outputs, norm_conditioning)
      return outputs
    if self._remat:
      call_function = hk.remat(call_function)
    return call_function(inputs, norm_conditioning)


class NormalizationLayer(hk.Module):
  """An activation normalization layer supporting norm conditioning."""

  @hk.name_like("__call__")
  def __init__(
      self,
      *,
      name: str,
      activation_normalization: str,
      use_norm_conditioning: bool = False,
      group_norm_groups: Optional[int] = None,
      stack: bool = False,
      ):
    """Inits the module.

    Args:
      name: Name of the module.
      activation_normalization: Type of normalization activation to use. One of
        "layer_norm", "rms_norm", "group_norm",
      use_norm_conditioning: Whether to use norm conditioning, instead of
        learned scale for the normalization module. If True,
        `norm_conditioning` must be passed to the `call` method.
      group_norm_groups: Number of groups for "group_norm".
      stack: If True, use stacked layers.
    """
    super().__init__(name=name)

    if use_norm_conditioning:
      # If using norm conditioning, it is no longer the responsibility of the
      # normalization module itself (e.g. LayerNorm) to learn its scale and
      # offset. These will be learned for the module by the norm conditioning
      # layer instead.
      create_scale = create_offset = False
    else:
      create_scale = create_offset = True

    # Note: if norm conditioning is being used, name will not appear in the
    # network's parameters since the normalization layer no longer has
    # learnable parameters.
    if activation_normalization == "layer_norm":
      self._normalization = hk.LayerNorm(
          axis=-1,
          create_scale=create_scale,
          create_offset=create_offset,
          name=activation_normalization,
      )
    elif activation_normalization == "rms_norm":
      self._normalization = hk.RMSNorm(
          axis=-1, create_scale=create_scale, name=activation_normalization
      )
    elif activation_normalization == "group_norm":
      self._normalization = hk.GroupNorm(
          groups=group_norm_groups,
          axis=-1,
          create_scale=create_scale,
          create_offset=create_offset,
          name=activation_normalization,
      )
    else:
      raise ValueError(
          "Unsupported activation normalization type: "
          f"{activation_normalization}"
      )

    self._norm_conditioning_layer = None
    if use_norm_conditioning:
      self._norm_conditioning_layer = LinearNormConditioning(
          name="linear_norm_conditioning",
          stack=stack)
    else:
      self._norm_conditioning_layer = None

  def __call__(
      # TODO(alvarosg): Expose `is_training` for controlling dropout of the MLP.
      self, inputs: jax.Array, norm_conditioning: jax.Array | None = None):

    outputs = self._normalization(inputs)

    if self._norm_conditioning_layer is not None:
      if norm_conditioning is None:
        raise ValueError("Expecting `norm_conditioning` features.")
      outputs = self._norm_conditioning_layer(outputs, norm_conditioning)
    return outputs


def get_activation_fn(name: str) -> Callable[..., Any]:
  """Return activation function corresponding to function_name."""
  if name == "identity":
    return lambda x: x
  if hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  if hasattr(jnp, name):
    return getattr(jnp, name)
  raise ValueError(f"Unknown activation function {name} specified.")


class MLP(hk.Module):
  """A multi-layer perceptron module.

  Mostly forked from Haiku:
  * With optimizations added on top
  * For flexibility going forwards.
  * Stability of the implementation.
  * Changed name scope of the variables remove the unnecessary "~" via
  `name_like`.
  """

  @hk.name_like("__call__")
  def __init__(
      self,
      *,
      output_sizes: Iterable[int],
      drop_first_matmul: bool = False,
      w_init: hk.initializers.Initializer | None = None,
      b_init: hk.initializers.Initializer | None = None,
      with_bias: bool = True,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
      activate_final: bool = False,
      name: str | None = None,
      stack: bool = False,
  ):
    """Constructs an MLP.

    Args:
      output_sizes: Sequence of layer sizes.
      drop_first_matmul: If True, the first matmul will be dropped.
      w_init: Initializer for :class:`~haiku.Linear` weights.
      b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
        ``with_bias=False``.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.
      stack: If True, use stacked layers.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init
    self.activation = activation
    self.activate_final = activate_final
    layers = []
    output_sizes = tuple(output_sizes)
    linear_cls = StackedLinear if stack else hk.Linear
    linear_name_prefix = "stacked_linear" if stack else "linear"
    for index, output_size in enumerate(output_sizes):
      if drop_first_matmul and index == 0:
        if with_bias:
          if stack:
            bias_dims = [-2, -1] if stack else [-1]
            if b_init is not None:
              # In the none case, b_init is zeros so we don't need to use the
              # stacked init wrapper.
              b_init = stacked_init_wrapper(b_init)
          else:
            bias_dims = [-1]
          # Use "linear" for the name for easier parametersurgery.
          layer = hk.Bias(
              bias_dims=bias_dims, b_init=b_init, name=f"{linear_name_prefix}_0"
          )
        else:
          layer = lambda x: x
      else:
        layer = linear_cls(
            output_size=output_size,
            w_init=w_init,
            b_init=b_init,
            with_bias=with_bias,
            name=f"{linear_name_prefix}_{index}",
        )
      layers.append(layer)
    self.layers = tuple(layers)
    self.output_size = output_sizes[-1] if output_sizes else None

  def __call__(
      self,
      inputs: jax.Array,
      dropout_rate: float | None = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape ``[batch_size, input_size]``.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.

    Returns:
      The output of the model of size ``[batch_size, output_size]``.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = hk.PRNGSequence(rng) if rng is not None else None  # pyrefly: ignore[bad-assignment]
    num_layers = len(self.layers)

    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (num_layers - 1) or self.activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          out = hk.dropout(next(rng), dropout_rate, out)  # pyrefly: ignore[bad-argument-type]
        out = self.activation(out)

    return out


class LinearNormConditioning(hk.Module):
  """Module for norm conditioning.

  Conditions the normalization of "inputs" by applying a linear layer to the
  "norm_conditioning" which produces the scale and variance which are applied to
  each channel (across the last dim) of "inputs".
  """

  def __init__(self, name="linear_norm_conditioning", stack: bool = False):
    super().__init__(name=name)
    self._stack = stack

  def __call__(self, inputs: jax.Array, norm_conditioning: jax.Array):
    feature_size = inputs.shape[-1]
    if self._stack:
      # This must match rather than be broadcastable to ensure that the
      # conditioning weight has a matching stack size.
      if norm_conditioning.shape[-2] != inputs.shape[-2]:
        raise ValueError(
            "Norm conditioning stack size does not match input stack size."
        )
      conditional_linear_layer = StackedLinear(
          output_size=2 * feature_size,
          w_init=hk.initializers.TruncatedNormal(stddev=1e-8),
      )
    else:
      conditional_linear_layer = hk.Linear(
          output_size=2 * feature_size,
          w_init=hk.initializers.TruncatedNormal(stddev=1e-8),
      )
    conditional_scale_offset = conditional_linear_layer(norm_conditioning)
    scale_minus_one, offset = jnp.split(conditional_scale_offset, 2, axis=-1)
    scale = scale_minus_one + 1.0
    return inputs * scale + offset


def summed_args(
    update: Optional[Callable[..., jraph.ArrayTree]] = None,
) -> Union[
    Callable[..., jraph.ArrayTree],
    Callable[[Callable[..., jraph.ArrayTree]], jraph.ArrayTree],
]:
  """Decorator that sums arguments before being passed to an update_fn.

  This is based on the function jraph.concatenateed_args.

  By default node, edge and global features are passed separately to update
  functions. However, in some specific cases we may want to sum them together.
  This wrapper sums the arguments for you.

  Args:
    update: an update function that takes ``jnp.ndarray``.

  Returns:
    A wrapped function with the arguments summed.
  """

  def _decorate(f):

    @functools.wraps(update)  # pyrefly: ignore[bad-argument-type]
    def wrapper(*args, **kwargs):
      combined_args = tree.tree_flatten(args)[0] + tree.tree_flatten(kwargs)[0]
      return f(sum(combined_args))

    return wrapper

  # If the update function is passed, then decorate the update function.
  if update:
    return _decorate(update)

  # Otherwise, return the decorator.
  return _decorate


class StackedLinear(hk.Module):
  """Linear module for stacked features."""

  def __init__(
      self,
      output_size: int,
      with_bias: bool = True,
      w_init: hk.initializers.Initializer | None = None,
      b_init: hk.initializers.Initializer | None = None,
      name: str | None = None,
  ):
    """Constructs a Linear module for stacked features.

    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: See, hk.Linear.
      b_init: See, hk.Linear.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros

  def __call__(
      self,
      inputs: jax.Array,
      *,
      stack_size: int | None = None,
      precision: lax.Precision | None = None,
  ) -> jax.Array:
    """Computes a linear transform of the input.

    Args:
      inputs: The input array.
      stack_size: Optional stack size. If provided, it must be broadcastable
        to the input stack size.
      precision: Optional precision for the matmul.

    Returns:
      The output of the linear layer.
    """
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    output_shape = list(inputs.shape)
    output_shape[-1] = self.output_size
    input_size = self.input_size = inputs.shape[-1]
    if stack_size:
      if not (stack_size == inputs.shape[-2] or stack_size == 1):
        raise ValueError(
            f"Stack size {stack_size} is not broadcastable to input stack size"
            f" {inputs.shape[-2]}."
        )
    else:
      stack_size = inputs.shape[-2]

    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      stddev = 1. / np.sqrt(self.input_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w_init = stacked_init_wrapper(w_init)

    b_init = self.b_init
    if b_init is not None:
      # If b_init is None, it will use zeros, so we don't need to wrap it.
      b_init = stacked_init_wrapper(b_init)
    w = hk.get_parameter(
        "w", [stack_size, input_size, self.output_size], dtype, init=w_init
    )

    out = jnp.einsum("...si,sij->...sj", inputs, w, precision=precision)

    if self.with_bias:
      b = hk.get_parameter(
          "b", [stack_size, self.output_size], dtype, init=b_init
      )
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    assert list(out.shape) == output_shape
    return out


def stacked_init_wrapper(init_fn):
  """Wraps an initializer function to work with stacked parameters.

  This wrapper takes an initializer function designed for unstacked parameters
  and adapts it to generate parameters for stacked modules. It assumes the
  stack dimension is the first dimension of the shape.

  Args:
    init_fn: An Haiku initializer function (e.g.,
      hk.initializers.TruncatedNormal).

  Returns:
    A wrapped initializer function that can be used for stacked parameters.
  """

  def wrapped_fn(shape, dtype):
    stack_size = shape[0]
    unstacked_shape = shape[1:]
    return jnp.stack(
        [init_fn(unstacked_shape, dtype) for _ in range(stack_size)]
    )

  return wrapped_fn
