import abc

from typing import Tuple, Optional

from graphcast import losses
from graphcast import xarray_jax
import jax.numpy as jnp
import xarray
import chex

StackedLossAndDiagnostics = losses.StackedLossAndDiagnostics


class StackedPredictor(abc.ABC):
  """A possibly-trainable predictor of weather, exposing an xarray-based API.

  Typically wraps an underlying JAX model and handles translating the xarray
  Dataset values to and from plain JAX arrays that are convenient for input to
  (and output from) the underlying model.

  Different subclasses may exist to wrap different kinds of underlying model,
  e.g. models taking stacked inputs/outputs, models taking separate 2D and 3D
  inputs/outputs, autoregressive models.

  You can also implement a specific model directly as a Predictor if you want,
  for example if it has quite specific/unique requirements for its input/output
  or loss function, or if it's convenient to implement directly using xarray.
  """

  @abc.abstractmethod
  def __call__(self,
               inputs: chex.Array,
               **optional_kwargs
               ) -> chex.Array:
    """Makes predictions.

    This is only used by the Experiment for inference / evaluation, with
    training going via the .loss method. So it should default to making
    predictions for evaluation, although you can also support making predictions
    for use in the loss via an is_training argument -- see
    LossFunctionPredictor which helps with that.

    Args:
      inputs: A chex.Array of inputs and forcings.
      **optional_kwargs: Implementations may support extra optional kwargs,
        provided they set appropriate defaults for them.

    Returns:
      Predictions, as a chex.Array
      For probabilistic predictors which can return multiple samples from a
      predictive distribution, these should (by convention) be returned along
      an additional 'sample' dimension.
    """

  def loss(self,
           inputs: chex.Array,
           targets: chex.Array,
           **optional_kwargs,
           ) -> StackedLossAndDiagnostics:
    """Computes a training loss, for predictors that are trainable.

    Why make this the Predictor's responsibility, rather than letting callers
    compute their own loss function using predictions obtained from
    Predictor.__call__?

    Doing it this way gives Predictors more control over their training setup.
    For example, some predictors may wish to train using different targets to
    the ones they predict at evaluation time -- perhaps different lead times and
    variables, perhaps training to predict transformed versions of targets
    where the transform needs to be inverted at evaluation time, etc.

    It's also necessary for generative models (VAEs, GANs, ...) where the
    training loss is more complex and isn't expressible as a parameter-free
    function of predictions and targets.

    Args:
      inputs: An chex.Array.
      **optional_kwargs: Implementations may support extra optional kwargs,
        provided they set appropriate defaults for them.

    Returns:
      loss: A DataArray with dimensions ('batch',) containing losses for each
        element of the batch. These will be averaged to give the final
        loss, locally and across replicas.
      diagnostics: Mapping of additional quantities to log by name alongside the
        loss. These will will typically correspond to terms in the loss. They
        should also have dimensions ('batch',) and will be averaged over the
        batch before logging.
        You need not include the loss itself in this dict; it will be added for
        you.
    """
    del targets, forcings, optional_kwargs
    batch_size = inputs.sizes['batch']
    dummy_loss = xarray_jax.DataArray(jnp.zeros(batch_size), dims=('batch',))
    return dummy_loss, {}  # pytype: disable=bad-return-type

  def loss_and_predictions(
      self,
      inputs: chex.Array,
      targets: chex.Array,
      **optional_kwargs,
      ) -> Tuple[StackedLossAndDiagnostics, chex.Array]:
    """Like .loss but also returns corresponding predictions.

    Implementing this is optional as it's not used directly by the Experiment,
    but it is required by autoregressive.Predictor when applying an inner
    Predictor autoregressively at training time; we need a loss at each step but
    also predictions to feed back in for the next step.

    Note the loss itself may not be directly regressing the predictions towards
    targets, the loss may be computed in terms of transformed predictions and
    targets (or in some other way). For this reason we can't always cleanly
    separate this into step 1: get predictions, step 2: compute loss from them,
    hence the need for this combined method.

    Args:
      inputs:
      targets:
      **optional_kwargs:
        As for self.loss.

    Returns:
      (loss, diagnostics)
        As for self.loss
      predictions:
        The predictions which the loss relates to. These should be of the same
        shape as what you would get from
        `self.__call__(inputs, targets_template=targets)`, and should be in the
        same 'domain' as the inputs (i.e. they shouldn't be transformed
        differently to how the predictor expects its inputs).
    """
    raise NotImplementedError
